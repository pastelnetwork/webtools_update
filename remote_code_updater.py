#!/usr/bin/env python3
# Copyright (c) 2022-2023 The Pastel Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://www.opensource.org/licenses/mit-license.php.
import sys
import argparse
import requests
import hashlib
from datetime import datetime
import threading
from enum import IntEnum
import importlib.util
from urllib.parse import urljoin, urlparse
from http import HTTPStatus
from pathlib import Path
import shutil
from nacl.public import PrivateKey
from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError
import logging

PRIVATE_KEY_FILENAME: str = 'private_key.txt'
VERIFY_KEY_FILENAME: str = 'verify_key.txt'
SIGNATURE_FILENAME: str = 'signature.txt'

class RemoteCodeUpdater:
    """Remote code updater.
       This class can be used in two modes:
       1) creating signature for the given python file
        execute [remote_code_updater.py filename], where 
            filename - absolute or relative filename to be signed
        privKey.txt file should exist in the same folder as 'filename'
        signature.txt file is created in the same folder as 'filename'
       2) updating existing python module in run-time
    """
    # 15 mins wait time while file is inuse
    INUSE_WAIT_TIME_SECS: float = 900
    
    class UpdateStatus(IntEnum):
        NOT_CHANGED = 0
        UPDATED = 1
        UPDATED_NEED_RELOAD = 2
        UPDATE_DISABLED = 3
        ROLLED_BACK = 4
        FAILED_LOCAL_FILE_NOT_FOUND = -1
        FAILED_GET_SIGNATURE = -2
        FAILED_CALC_LOCAL_HASH = -3
        FAILED_GET_PUBLIC_KEY = -4
        FAILED_BACKUP = -5
        FAILED_UPDATE_FILE = -6
        FAILED_SINGATURE_INVALID = -7
        FAILED_UNKNOWN_EXCEPTION = -8
        FAILED_ROLLBACK_FILE = -9
    
    
    def __init__(self, temp_path: Path, update_url_root_path: str, original_file: Path, logger: logging.Logger = None):
        self.verify_key = None
        self.temp_path: Path = temp_path
        self.update_url_root_path: str = update_url_root_path
        self.original_file: Path = original_file
        self.backup_file: Path = None
        self.is_module_file_updated: bool = False
        self.filename = None
        if self.original_file.exists():
            self.filename = self.original_file.name
        self.inuse_ref_counter: int = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        if not logger:
            # initialize log for standalone execution
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-5s] %(message)s')
            logger = logging.getLogger("RCU")
            logger.propagate = False
            logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger = logger
        self._is_update_disabled: bool = False
        

    def get_url(self, filename) -> str:
        return urljoin(self.update_url_root_path, filename)
        
        
    def disable_update(self):
        """Disable remote update for the module.
        """
        if self.filename:
            self.logger.info(f"Remote update is disabled for '{self.filename}'")
        self._is_update_disabled = True
        
        
    @property
    def url_verify_key(self) -> str:
        return self.get_url(VERIFY_KEY_FILENAME)
    
    
    @property
    def url_signature(self) -> str:
        return self.get_url(SIGNATURE_FILENAME)
    
    
    @property
    def temp_filename(self) -> str:
        fn = Path(self.filename)
        return f'{fn.stem}_downloaded{fn.suffix}'
    
    
    @staticmethod
    def retrieve_resource_data(url: str) -> str:
        # check if resource is url or local path
        if urlparse(url).scheme:
            r = requests.get(url)
            if r.status_code != HTTPStatus.OK:
                raise RuntimeError(f'HTTP Status code: {r.status_code}')
            signature_data = r.text
        else:
            signature_data = Path(url).expanduser().read_text()
        return signature_data

    
    def get_verify_key(self) -> hex:
        """Retrieve verify key from the remote location.
        """
        verify_key_hex = None
        try:
            verify_key_hex = self.retrieve_resource_data(self.url_verify_key)
        except Exception as exc:
            self.logger.exception(f'Failed to retrieve verify key file for {self.filename}')
        else:    
            self.logger.info(f'Got the verify key from {self.url_verify_key}')
        return verify_key_hex

    
    def get_latest_version(self, is_use_local: bool = False) -> str:
        """Calculate sha3 hash of the code file.

        Args:
            is_use_local (bool, optional): whether to load code file from local file system. Defaults to False.

        Returns:
            str: sha3-256 hash of the code file
        """
        file_data = ''
        if is_use_local:
            file_data = self.original_file.read_text()
        else:
            try:
                file_data = self.retrieve_resource_data(self.get_url(self.filename))
            except Exception as exc:
                self.logger.exception(f'Failed to retrieve latest version of code file {self.filename}')
                return None
            else:
                self.logger.info(f"Got the latest version of code file '{self.filename}'")                
            self.temp_file = self.temp_path / self.temp_filename
            self.temp_file.write_text(file_data)
            if not self.temp_file.exists():
                self.logger.error(f"Failed to save '{self.filename}' to temporary file {str(self.temp_file)}")
                return None
        sha3_256_hash = hashlib.sha3_256(file_data.encode('utf-8')).hexdigest()
        self.logger.info(f'SHA3-256 hash of the code file: {sha3_256_hash}')
        return sha3_256_hash
       

    def verify_signature_and_update(self) -> UpdateStatus:
        if not self.original_file.exists():
            self.logger.error(f"Local file to update not found '{str(self.original_file)}'")
            return self.UpdateStatus.FAILED_LOCAL_FILE_NOT_FOUND
        file_data = self.original_file.read_text().encode("utf-8")
        sha3_256_hash_of_code = hashlib.sha3_256(file_data).hexdigest()
        
        # check if signature is url or local path
        signature_hex = ''
        try:
            signature_hex = self.retrieve_resource_data(self.url_signature)
        except Exception as exc:
            self.logger.exception(f"Failed to retrieve signature for file '{self.filename}' from {self.url_signature}")
            return self.UpdateStatus.FAILED_GET_SIGNATURE
        else:
            self.logger.info(f"Got the signature for '{self.filename}' from {self.url_signature}:\n{signature_hex}")
        signature = bytes.fromhex(signature_hex)
        sha3_256_hash_of_code_new = self.get_latest_version()
        if sha3_256_hash_of_code_new is None:
            return self.UpdateStatus.FAILED_CALC_LOCAL_HASH
        
        if sha3_256_hash_of_code == sha3_256_hash_of_code_new:
            self.logger.info(f"'{self.filename}' is the latest version")
            return self.UpdateStatus.NOT_CHANGED
        
        verify_key_hex = self.get_verify_key()
        if not verify_key_hex:
            self.logger.info(f"Cannot retrieve verify key for '{self.filename}'")
            return self.UpdateStatus.FAILED_GET_PUBLIC_KEY
            
        verify_key = VerifyKey(verify_key_hex, encoder=HexEncoder)
        try:
            # Verify the signature:
            self.logger.info(f'Signature length: {len(signature)}')
            verify_key.verify(sha3_256_hash_of_code_new.encode('utf-8'), signature)
            self.logger.info(f"The signature is valid for '{self.filename}'")
            # generate backup file name in format: {stem}_{timestamp}{suffix}
            self.backup_file = Path(self.temp_path) / f"{self.original_file.stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{self.original_file.suffix}"
            # Since the signature is valid, we can replace the old version of the file with the new one (backing up the old one first):
            try:
                shutil.copyfile(self.original_file, self.backup_file)
            except Exception as exc:
                self.logger.exception(f'Could not create a backup of the original [{str(self.original_file)}] file to [{str(self.backup_file)}]')
                self.backup_file = None
                return self.UpdateStatus.FAILED_BACKUP
            else:
                self.logger.info(f"File '{str(self.original_file)}' backed up to [{str(self.backup_file)}]")
                            
            try:
                shutil.copyfile(str(self.temp_file), str(self.original_file))
            except Exception as exc:
                self.logger.error(f"Could not replace the old version of '{str(self.original_file)}' with the new one. {exc!r}")
                return self.UpdateStatus.FAILED_UPDATE_FILE
            else:
                self.logger.info(f"'{self.filename}' was updated to the latest version")
                self.is_module_file_updated = True
                            
        except BadSignatureError:
            self.logger.info(f'The signature for {self.filename} is invalid')
            return self.UpdateStatus.FAILED_SINGATURE_INVALID
        except Exception as exc:
            self.logger.exception(f"Exception occurred while updating '{self.filename}'")
            return self.UpdateStatus.FAILED_UNKNOWN_EXCEPTION
        
        return self.UpdateStatus.UPDATED


    def can_update_module(self) -> bool:
        return self.inuse_ref_counter == 0
        

    def release_module(self):
        with self._cv:
            if self.inuse_ref_counter > 0:
                self.inuse_ref_counter -= 1
                self.logger.info(f"'{self.filename}' inuse counter released [{self.inuse_ref_counter}]")
            if self.inuse_ref_counter == 0:
                self._cv.notify_all()

    def _increment_inuse_counter(self):
        self.inuse_ref_counter += 1
        self.logger.info(f"'{self.filename}' inuse counter incremented [{self.inuse_ref_counter}]")
        

    def wait_while_inuse(self) -> bool:
        with self._cv:
            self.logger.info(f"Waiting for 0 inuse counter for '{self.filename}' to reload it")
            can_update = self._cv.wait_for(self.can_update_module, self.INUSE_WAIT_TIME_SECS)
            if not can_update:
                return False
            self._increment_inuse_counter()
        return True


    def refresh(self) -> bool:
        is_updated: bool = False
        self._lock.acquire()
        try:
            spec = importlib.util.find_spec(self.original_file.stem, str(self.original_file))
            if self.is_module_file_updated:
                status = self.UpdateStatus.UPDATED_NEED_RELOAD
            elif self._is_update_disabled:
                status = self.UpdateStatus.UPDATE_DISABLED
            else:
                status = self.verify_signature_and_update()
            need_update = False
            module_update = False
            if spec is None:
                spec = importlib.util.spec_from_file_location(self.original_file.stem, str(self.original_file))
                module = importlib.util.module_from_spec(spec)
                need_update = True
            else:
                module = importlib.util.module_from_spec(spec)
                is_module_imported = sys.modules.get(module.__name__) is not None
                if status in [self.UpdateStatus.UPDATED, self.UpdateStatus.UPDATED_NEED_RELOAD] or not is_module_imported:
                    module_update = True
                    # check if module is in use right now, wait until it becomes not used
                    if not self.wait_while_inuse():
                        # timed out - have to skip update
                        self.logger.warning(f"Timeout {int(self.INUSE_WAIT_TIME_SECS)} secs elapsed while waiting for '{self.filename}' inuse 0 counter")
                    else:
                        need_update = True
                elif status == self.UpdateStatus.NOT_CHANGED:
                    with self._cv:
                        self._increment_inuse_counter()
                elif status < 0:
                    self.logger.error(f"Failed to update '{self.filename}'. {status.name}")
            if need_update:
                sys.modules[module.__name__] = module
                spec.loader.exec_module(module)
                if module_update or self._is_update_disabled or status < 0:
                    update_msg = 'has been imported'
                else:
                    update_msg = 'is successfully updated and reloaded'
                self.logger.info(f"Module '{module.__name__}' {update_msg}")
                is_updated = True
                # reset file updated flag
                self.is_module_file_updated = False
        except Exception:
            self.logger.exception(f"Failed to update '{self.filename}'")
        finally:
            self._lock.release()
        return is_updated
                   
    def rollback_module(self) -> int:
        if self._is_update_disabled:
            return self.UpdateStatus.UPDATE_DISABLED
        if self.backup_file is None:
            self.logger.error(f"Backup file is not created for '{self.filename}'")
            return self.UpdateStatus.FAILED_ROLLBACK_FILE
        try:
            shutil.copyfile(str(self.backup_file), str(self.original_file))
            self.logger.info(f"'{self.filename}' was updated from backup version '{str(self.backup_file)}'")
        except Exception as exc:
            self.logger.exception(f"Could not replace the old version of '{str(self.original_file)}' with the new one")
            return self.UpdateStatus.FAILED_ROLLBACK_FILE
        else:
            self.is_module_file_updated = True
            self.backup_file = None
        return self.UpdateStatus.ROLLED_BACK

    def create_signature(self):
        self.logger.info(f"Creating signature for the file '{str(self.original_file)}'")
        
        # load private & verify keys:
        root_dir = Path(self.original_file.parent)
        private_key_file = root_dir / PRIVATE_KEY_FILENAME
        verify_key_file = root_dir / VERIFY_KEY_FILENAME
        
        try:
            private_key_bytes = bytes.fromhex(private_key_file.read_text())
        except BaseException:
            self.logger.exception(f"Failed to load private key from file '{str(private_key_file)}'")
            return
        else:
            self.logger.info('Private key loaded successfully')
            
        try:
            verify_key_bytes = bytes.fromhex(verify_key_file.read_text())
        except BaseException:
            self.logger.exception(f"Failed to load verify key from file '{str(verify_key_file)}'")
            return
        else:
            self.logger.info('Verify key loaded successfully')
        
        signing_key = SigningKey(private_key_bytes)
        sha3_256_hash = self.get_latest_version(True)
        # sign the hash:
        signed_msg = signing_key.sign(sha3_256_hash.encode('utf-8'))
        signature = signed_msg.signature
        # save signature to file:
        signature_hex_digest = signature.hex()
        self.logger.info(f'Signature [len={len(signature)}]: {str(signature_hex_digest)}')
        signature_file = Path(self.original_file.parent) / SIGNATURE_FILENAME
        signature_file.write_text(signature_hex_digest)
        self.logger.info(f"File '{self.original_file.absolute()}' was signed successfully.\nSignature created in '{signature_file.absolute()}'")
        
        # Create a VerifyKey object from a hex serialized public key
        verify_key = VerifyKey(verify_key_bytes)
        
        # validate signed hash
        try:
            verify_key.verify(signed_msg.message, signature)
        except BadSignatureError:
            self.logger.exception("Signature verification failed !!!")
        else:
            self.logger.info("Signature verified successfully")           
            
            
def generate_signature_key():
    # Generate new private key
    private_key = PrivateKey.generate()
    private_key_bytes = private_key.encode()
    private_key_hex = private_key.encode(encoder=HexEncoder)
    
    # Generate signing key with private key as a seed
    signing_key = SigningKey(private_key_bytes)
    
    # Obtain the verify key for a given signing key
    verify_key = signing_key.verify_key
    verify_key_bytes = verify_key.encode()
    verify_key_hex = verify_key.encode(encoder=HexEncoder)
    
    # save keys
    print(f"Private key ({private_key.SIZE} bytes):\n{private_key_hex}")
    print(f"Verify key ({len(verify_key_bytes)} bytes):\n{verify_key_hex}")
    Path(PRIVATE_KEY_FILENAME).write_bytes(private_key_hex)
    Path(VERIFY_KEY_FILENAME).write_bytes(verify_key_hex)
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create signature for the given file, to be verified on remote update')
    parser.add_argument('-k', '--newkey', action="store_true", help='generate new signing/verify key pair')
    parser.add_argument('-s', '--sign', action="store", help='create signature for the given file')
    options = parser.parse_args()

    if options.newkey:
        generate_signature_key()
    elif options.sign:
        file_to_update = Path(options.sign)
        if not file_to_update.exists():
            raise FileNotFoundError(f'File not found [{file_to_update.absolute()}]')
        RemoteCodeUpdater('', '', file_to_update).create_signature()
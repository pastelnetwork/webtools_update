# Copyright (c) 2021-2023 The Pastel Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://www.opensource.org/licenses/mit-license.php.
from typing import Dict, Tuple, List, Optional
import os
import time
import socket
import shutil
import urllib.request
import re
import random
import json
import gc
import asyncio
import nest_asyncio
nest_asyncio.apply() # patch asyncio to allow nested event loops
import httpx
import threading
from urllib.parse import quote
import threading
from datetime import datetime, timedelta
import datefinder
from html.parser import HTMLParser
from contextlib import contextmanager
import _thread

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys as SeleniumKeys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import (
    NoSuchElementException,
    NoAlertPresentException,
    MoveTargetOutOfBoundsException,
)
import pandas as pd
import pyimgur
import numpy as np
from bs4 import BeautifulSoup

from config import (
    CONFIG,
    CHROME_DEVTOOLS_PORT_RANGE_START,
    CHROME_DEVTOOLS_PORT_RANGE_END,
)
from dd_logger import logger
from utils import (
    Timer,
    NumpyEncoder,
    get_sha3_256_from_data,
    get_hash_sha3_256_from_file,
    is_in_debugger,
    compress_text_data_with_zstd_and_encode_as_base64,
    get_free_port_from_range,
    walk_child_processes,
    cleanup_old_files,
    kill_process,
    kill_process_tree,
    is_port_in_use,
    cleanup_chrome_user_data_dir,
)
from dupe_detection_params import (
    DupeDetectionTaskParams,
    TaskCancelledError,
)
from image_utils import (
    DDImage,
    ImageDataPreProcessedBase64String,
    ImageDataPreProcessedUrl,
    filter_out_dissimilar_images_batch,
    sync__get_image_url_as_base64_string,
    extract_base64_data_from_image_url,
    resample_img_src,
    get_image_url_data,
    validate_url_load_time,
)

CLIENT_ID = "689300e61c28cc7"
CLIENT_SECRET = "6c45e31ca3201a2d8ee6709d99b76d249615a10c"
im = pyimgur.Imgur(CLIENT_ID, CLIENT_SECRET)

WEBTOOLS_VERSION = "1.23"
DEBUG_TIME_LIMIT_SECS = 900
METADATA_DOWNLOAD_TIMEOUT_SECS = 120
METADATA_DOWNLOAD_MAX_WORKERS = 15
METADATA_URL_LOAD_TIMEOUT_SECS = 15
METADATA_MIN_IMAGE_SIZE_TO_RETRIEVE_BYTES = 5000
MAX_SEARCH_RESULTS = 40
MAX_RESULTS_TO_RETURN = 15
GOOGLE_LENS_RESULT_PAGE_TIMEOUT = 35

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg
        

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

class MyHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        self.data = []
        super(MyHTMLParser, self).__init__(*args, **kwargs)

    def handle_data(self, data):
        data = data.strip()
        if data:
            self.data.append(data)


def remove_dupes_from_list_but_preserve_order_func(list_of_items):
    deduplicated_list = list(dict.fromkeys(list_of_items).keys())
    return deduplicated_list


def flatten_func(list_of_lists):
    if type(list_of_lists) != bool:
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return flatten_func(list_of_lists[0]) + flatten_func(list_of_lists[1:])
        return list_of_lists[:1] + flatten_func(list_of_lists[1:])
    else:
        return list_of_lists


def in_nested_list_func(my_list, item):
    if item in my_list:
        return True
    else:
        return any(in_nested_list_func(sublist, item) for sublist in my_list if isinstance(sublist, list))


def generate_rare_on_internet_graph_func(combined_summary_df, rare_on_internet__similarity_df, rare_on_internet__adjacency_df):
    links_list = []
    nodes_list = []
    list_of_result_titles = combined_summary_df['title'].values.tolist()
    list_of_search_result_rankings = [x+1 for x in combined_summary_df.index.tolist()]
    list_of_img_src_strings = combined_summary_df['img_src_string'].values.tolist()
    list_of_image_resolutions = combined_summary_df['resolution_string'].values.tolist()
    list_of_misc_related_image_urls = combined_summary_df['misc_related_image_url'].values.tolist()
    list_of_misc_related_images_as_b64_strings = combined_summary_df['misc_related_image_as_b64_string'].values.tolist() 
    list_of_link_urls_primary = combined_summary_df['original_url'].values.tolist()
    list_of_dates = combined_summary_df['date_string'].values.tolist()
    list_of_image_labels = rare_on_internet__adjacency_df.index.tolist()
    image_label_to_result_title_dict = dict(zip(list_of_image_labels, list_of_result_titles))
    image_label_to_search_result_ranking_dict = dict(zip(list_of_image_labels, list_of_search_result_rankings))
    image_label_to_img_src_dict = dict(zip(list_of_image_labels, list_of_img_src_strings))
    image_label_to_image_resolutions_dict = dict(zip(list_of_image_labels, list_of_image_resolutions))
    image_label_to_link_urls_primary_dict = dict(zip(list_of_image_labels, list_of_link_urls_primary))
    image_label_to_dates_dict = dict(zip(list_of_image_labels, list_of_dates))
    image_label_to_misc_related_image_urls_dict = dict(zip(list_of_image_labels, list_of_misc_related_image_urls))
    image_label_to_misc_related_images_as_b64_strings_dict = dict(zip(list_of_image_labels, list_of_misc_related_images_as_b64_strings))   
    counter = 0
    for ii, label1 in enumerate(list_of_image_labels):
        try:
            nodes_list = nodes_list + [dict(id=ii,
                                        image_label=label1,
                                        title=image_label_to_result_title_dict[label1],
                                        search_result_ranking=image_label_to_search_result_ranking_dict[label1],
                                        img_src_string=image_label_to_img_src_dict[label1],
                                        resolution_string=image_label_to_image_resolutions_dict[label1],
                                        original_url=image_label_to_link_urls_primary_dict[label1],
                                        date_string=image_label_to_dates_dict[label1],
                                        misc_related_images_urls=image_label_to_misc_related_image_urls_dict[label1],
                                        misc_related_images_as_b64_strings=image_label_to_misc_related_images_as_b64_strings_dict[label1],
                                        )]
        except BaseException as e:
            logger.exception('Encountered error adding node to rare on internet graph structure')
        for jj, label2 in enumerate(list_of_image_labels):
            current_adjacency = rare_on_internet__adjacency_df.iloc[ii, jj]
            if current_adjacency and label1 != label2:
                try:
                    counter = counter + 1
                    links_list = links_list + [dict(source=ii, target=jj, connection_strength=str(rare_on_internet__similarity_df.loc[label1, label2]))]
                except BaseException as e:
                    logger.exception('Encountered error adding link to rare on internet graph structure')
    current_graph = {'nodes': nodes_list, 'links': links_list}
    #print('current_graph', current_graph)
    current_graph_json = json.dumps(current_graph, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    return current_graph_json


def generate_alt_rare_on_internet_graph_func(list_of_images_as_base64__filtered, list_of_image_src_strings__filtered, list_of_image_alt_strings__filtered, list_of_href_strings__filtered, alt_list_of_image_base64_hashes_filtered, alt_rare_on_internet__similarity_df, alt_rare_on_internet__adjacency_df):
    links_list = []
    nodes_list = []
    list_of_images_as_base64 = list_of_images_as_base64__filtered
    list_of_image_hashes = alt_list_of_image_base64_hashes_filtered
    list_of_img_src_strings = list_of_image_src_strings__filtered
    list_of_img_alt_strings = list_of_image_alt_strings__filtered
    list_of_href_strings = list_of_href_strings__filtered
    list_of_image_labels =  alt_rare_on_internet__adjacency_df.index.tolist()
    image_label_to_image_base64_string = dict(zip(list_of_image_labels, list_of_images_as_base64))
    image_label_to_image_hash_dict = dict(zip(list_of_image_labels, list_of_image_hashes))
    image_label_to_img_src_dict = dict(zip(list_of_image_labels, list_of_img_src_strings))
    image_label_to_img_alt_dict = dict(zip(list_of_image_labels, list_of_img_alt_strings))
    image_label_to_img_original_link_dict = dict(zip(list_of_image_labels, list_of_href_strings))

    cntr = 0
    for ii, label1 in enumerate(list_of_image_labels):
        try:
            nodes_list = nodes_list + [dict(id=ii,
                                            image_label=label1,
                                            image_base64_string = image_label_to_image_base64_string[label1],
                                            sha3_256_hash_of_image_base64_string=image_label_to_image_hash_dict[label1],
                                            img_src=image_label_to_img_src_dict[label1],
                                            img_alt=image_label_to_img_alt_dict[label1],
                                            original_url=image_label_to_img_original_link_dict[label1]
                                            )]
        except BaseException as e:
            logger.exception('Encountered error adding node to alternative rare on internet graph structure')
        for jj, label2 in enumerate(list_of_image_labels):
            current_adjacency = alt_rare_on_internet__adjacency_df.iloc[ii, jj]
            if current_adjacency and label1 != label2:
                try:
                    cntr += 1
                    links_list = links_list + [dict(source=ii, target=jj, connection_strength=str(alt_rare_on_internet__similarity_df.loc[label1, label2]))]
                except BaseException as e:
                    logger.exception('Encountered error adding link to alternative rare on internet graph structure')
    current_graph = {'nodes': nodes_list, 'links': links_list}
    current_graph_json = json.dumps(current_graph, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    return current_graph_json


async def get_all_images_on_page_as_base64_encoded_strings(get_task_id: int, html_of_page: str,
                                                     min_image_size_to_retrieve: int = METADATA_MIN_IMAGE_SIZE_TO_RETRIEVE_BYTES,
                                                     timeout_in_secs: float = METADATA_DOWNLOAD_TIMEOUT_SECS,
                                                     max_results_to_collect: int = MAX_SEARCH_RESULTS) -> Tuple[List[str], List[str]]:
    soup = BeautifulSoup(html_of_page, "lxml")
    img_elements = soup.find_all("img")
    
    image_urls = { img_src for img in img_elements if (img_src := img.get("src")) and img_src.startswith(("http://", "https://")) }
    logger.info(f' [{get_task_id}] found {len(image_urls)} distinct img elements on the page')
    if not image_urls:
        return [], []

    list_of_base64_encoded_images = []
    list_of_corresponsing_image_urls = []
    
    timer = Timer(start_timer=True)
    httpx_timeout_config = httpx.Timeout((5.0, timeout_in_secs))  # connect=5.0, read=timeout_in_secs
    total_image_count = len(image_urls)
    is_max_results_reached = False
    is_timeout = False
    async with httpx.AsyncClient(timeout=httpx_timeout_config) as client:
        chunk_size = 10
        for i in range(0, len(image_urls), chunk_size):
            if timer.elapsed_time >= timeout_in_secs:
                logger.info(f' [{get_task_id}] timeout of {timeout_in_secs} seconds reached. Stopping image retrieval.')
                is_timeout = True
                break
            image_urls_chunk = {image_urls.pop() for _ in range(min(chunk_size, len(image_urls)))}
            
            tasks = [get_image_url_data(client, url, min_image_size_to_retrieve) for url in image_urls_chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
            for result in results:
                if isinstance(result, Exception):
                    continue
                base64_img_data, image_url = result
                if base64_img_data and image_url:
                    list_of_base64_encoded_images.append(base64_img_data)
                    list_of_corresponsing_image_urls.append(image_url)
                    if len(list_of_base64_encoded_images) >= max_results_to_collect:
                        is_max_results_reached = True
                        logger.info(f' [{get_task_id}] max results {max_results_to_collect} reached')
                        break
            logger.info(f' [{get_task_id}] processed batch [{i + 1}-{i + len(image_urls_chunk)}/{total_image_count}] with {len(image_urls_chunk)} URLs.'
                        f' Collected images: {len(list_of_base64_encoded_images)}')
            if is_max_results_reached:
                break
            
    if image_urls:  # URLs that were never processed
        urls_string = '\n'.join(image_urls)
        if is_max_results_reached:
            reason = f' due to reaching max results ({max_results_to_collect})'
        elif is_timeout:
            reason = f' due to timeout {timeout_in_secs} secs'
        else:
            reason = ''
        logger.info(f' [{get_task_id}] images ({len(image_urls)}) were not processed{reason}:\n{urls_string}')
            
    return list_of_base64_encoded_images, list_of_corresponsing_image_urls


def extract_corresponding_description_string_from_input_title_string_func(input_title_string, html_of_google_result_page):
    soup = BeautifulSoup(html_of_google_result_page, "lxml")
    try:
        list_of_found_title_strings = [str(x.string) for x in soup.find_all('h3')]
    except:
        list_of_found_title_strings = [""]   
    try:
        list_of_found_description_strings = [str(x.string) for x in soup.find_all('div') if x.has_attr('data-sncf')]
    except:
        list_of_found_description_strings = ["" for x in list_of_found_title_strings]
        
    if len(list_of_found_title_strings) > 0:
        for idx, current_found_title_string in enumerate(list_of_found_title_strings):
            input_title_string_trimmed = input_title_string[0:45]
            current_found_title_string_trimmed = current_found_title_string[0:45]
            if input_title_string_trimmed==current_found_title_string_trimmed:
                if len(list_of_found_description_strings) > idx:
                    return list_of_found_description_strings[idx]
    return ""


def extract_valid_dates_from_string_func(input_string):
    date_strings = datefinder.find_dates(text=input_string, strict=True)
    deduplicated_date_strings = remove_dupes_from_list_but_preserve_order_func(date_strings)
    list_of_date_strings_fixed = []
    for current_date in deduplicated_date_strings:
        if current_date > datetime.now() - timedelta(days=2):
            continue
        list_of_date_strings_fixed.append(current_date.isoformat().split('T0')[0].replace(' ','_').replace(':','_'))
    return list_of_date_strings_fixed


def get_fields_from_image_search_result(current_result):
    title = None
    original_url = None
    resolution = ''
    try:
        current_soup = BeautifulSoup(str(current_result), "lxml")
        label_element = current_soup.find("a", {"aria-label": True, "href": True})
        if label_element:
            title = label_element["aria-label"]
            original_url = label_element["href"]
            if title == '':
                div_subitem = label_element.find("div", {"data-item-title": True})
                if div_subitem:
                    title = div_subitem["data-item-title"]
                    if original_url is None or original_url == '':
                        original_url = div_subitem.get("data-action-url")
                        
        if original_url is None or original_url == '':
            original_url = current_soup.get("data-action-url")

        current_img_element = current_soup.find("img", {"aria-hidden": "true"})
        if current_img_element:
            img_url = current_img_element["src"]
            
        resolution_pattern = r"(\d+)x(\d+)"
        resolution_match = re.search(resolution_pattern, str(current_result))
        if resolution_match:
            resolution = resolution_match.group(0)
    except Exception as exc:
        logger.info(f"Error parsing image result: {str(exc)}")
    
    img_src = ''
    if img_url:
        try:
            img_src = sync__get_image_url_as_base64_string(img_url)
        except:
            logger.info(f'Could not retrieve image file from [{img_url}]')        
    if not title:
        title = ''
    if not original_url:
        original_url = ''
            
    return title, original_url, img_src, resolution


class ChromeDriver:
    
    def __init__(self, img: DDImage, dd_params: DupeDetectionTaskParams):
        self.dd_params = dd_params
        self.devtools_port = dd_params.chrome_devtools_port
        if is_port_in_use(self.devtools_port):
            logger.error(f'Chrome DevTools port {self.devtools_port} is already in use')
            self.devtools_port = get_free_port_from_range(CHROME_DEVTOOLS_PORT_RANGE_START + 100, CHROME_DEVTOOLS_PORT_RANGE_END)
        self.devtools_http_uri: str = f"http://localhost:{self.devtools_port}"
        self.devtools_ws_uri: str = None
        if dd_params.chromedriver_path is None or dd_params.chromedriver_path == '':
            raise ValueError('ChromeDriver path is empty')
        if dd_params.chrome_user_data_dir is None or dd_params.chrome_user_data_dir == '':
            raise ValueError('Chrome user data dir is empty')
        chrome_options = ChromeOptions()
        if not CONFIG.debug_chrome_driver_headless_mode:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1200")
        chrome_options.add_argument("--disable-gpu")
        # chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-subframe-process-reuse")
        chrome_options.add_argument("--dns-prefetch-disable")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument(f"--user-data-dir={dd_params.chrome_user_data_dir}")
        chrome_options.add_argument(f"--remote-debugging-port={self.devtools_port}")
        chrome_options.add_argument("--site-per-process")
        chrome_service_args = []
        if CONFIG.enable_chromedriver_logging:
            chrome_service_args.append("--verbose")
            chrome_service_args.append(f"--log-path={str(CONFIG.chromedriver_log_files_path / f'chromedriver_{os.getpid()}.log')}")
        prefs = {
            # specifies where to download files to by default
            "download.default_directory": str(CONFIG.internet_rareness_downloaded_images_path),
            #  whether we should ask the user if we should download a file (true) or just download it automatically
            "download.prompt_for_download": False,
            # if the download directory was changed by an upgrade from unsafe location to a safe location
            "download.directory_upgrade": True,
            "profile.default_content_setting_values.geolocation": 2,
            "profile.default_content_setting_values.media_stream_camera": 2,
            "profile.default_content_setting_values.media_stream_mic": 2,
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_setting_values.popups": 2,
            "profile.default_content_setting_values.automatic_downloads": 2,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        self.resized_image_save_path = None
        self.img = img
        logger.info(f'Initialized webtools v{WEBTOOLS_VERSION}')
        self.driver = None
        self.chrome_options = chrome_options
        self.chrome_service_args = chrome_service_args
        self.start_chrome()


    def start_chrome(self):
        logger.info(f'Starting ChromeDriver...')
        chrome_service = ChromeService(executable_path=self.dd_params.chromedriver_path,
                                       service_args=self.chrome_service_args)
        for i in range(3):
            try:
                self.driver = webdriver.Chrome(service=chrome_service,
                                               options=self.chrome_options)
                self.chromedriver_pid = self.driver.service.process.pid
                logger.info(f'Initialized ChromeDriver v{self.driver.capabilities["browserVersion"]}, pid={self.chromedriver_pid}')
                break
            except Exception as e:
                logger.exception(f'Error initializing ChromeDriver, attempt #{i + 1}. {e}')
                time.sleep(2)
        if self.driver is not None:
            # Set a timeout in secs for page load
            self.driver.set_page_load_timeout(10)
            # Set a timeout in secs for scripts
            self.driver.set_script_timeout(30)
        

    def __del__(self):
        self.close()        


    def check_task_cancelled(self):
        if self.dd_params.is_task_cancelled():
            raise TaskCancelledError(f"Dupe Detection Task [{self.dd_params.task_id}] cancelled.")

        
    def reopen(self, task_id: int, need_open_new_tab: bool = False):
        logger.info(f'[{task_id}] ChromeDriver hanged and will be reopened...')
        close_driver_thread = threading.Thread(target=self.close)
        close_driver_thread.start()
        close_driver_thread.join(timeout=10)
        
        #if close_driver_thread.is_alive():
        self.driver = None
        logger.info(f'[{task_id}] ChromeDriver close thread is still alive. Killing the process {self.chromedriver_pid}...')
        kill_process_tree(self.chromedriver_pid)
        cleanup_chrome_user_data_dir(self.dd_params.chrome_user_data_dir)
       
        self.start_chrome()
        if need_open_new_tab:
            self.open_new_tab(task_id)
            
               
    def close(self):
        if self.driver:
            try:
                logger.info('Closing ChromeDriver...')
                self.driver.quit()
                self.driver = None
            except:
                pass
       
       
    def wait_for_page_loaded(self, id: int = None, query_url: str = None, timeout_in_secs: int = 10):
        # wait for Ajax calls to complete
        log_prefix = f' [{id}]' if id else ''
        if self.is_jquery_loaded():
            try:
                logger.info(f'{log_prefix} waiting for jQuery Ajax calls to complete...')
                WebDriverWait(self.driver, timeout_in_secs).until(lambda wd: wd.execute_script("return jQuery.active == 0"))
                logger.info(f'{log_prefix} ...jQuery loaded')
            except TimeoutException:
                logger.info(f'{log_prefix} timed out ({timeout_in_secs} secs) waiting for jQuery Ajax calls to complete while getting metadata from [{query_url}]')
        # check and dismiss any alert dialogs
        try:
            alert = self.driver.switch_to.alert
            logger.info(f'{log_prefix} alert dialog found: {alert.text}')
            alert.dismiss()
        except NoAlertPresentException:
            pass
        logger.info(f'{log_prefix} waiting for the page to load...')
        try:
            WebDriverWait(self.driver, timeout_in_secs).until(lambda wd: wd.execute_script('return document.readyState') == 'complete')
        except TimeoutException as exc:
            logger.info(f'{log_prefix} timed out ({timeout_in_secs} secs) waiting for the page to load [{query_url}]')
            raise exc


    def driver_get_with_timeout(self, id: int, query_url: str, timeout_in_secs: int = 10):
        # Define a function that runs `driver.get` in a thread.
        def get_url_in_thread():
            self.driver.get(query_url)
            self.wait_for_page_loaded(id, query_url, timeout_in_secs)

        thread = threading.Thread(target=get_url_in_thread)
        thread.start()
        thread.join(timeout=timeout_in_secs)

        # If the thread is still active after timeout, raise an exception.
        if thread.is_alive():
            raise TimeoutException(f"Timeout ({timeout_in_secs} secs) exceeded for URL [{query_url}]")
        
       
    def driver_retrieve_url_data(self, id: int, query_url: str, timeout_in_secs: int = 10) -> str:
        self.driver_get_with_timeout(id, query_url, timeout_in_secs)
        page_source_data = self.driver.page_source
        logger.info(f' [{id}] ...page is loaded ({len(page_source_data)} bytes)!')
        return page_source_data


    def collect_renderer_ids(self) -> Dict[int, int]:
        """
        Collect all client IDs with their associated process IDs
        Returns:
            dict: {client_id: pid}
        """
        # Filter for renderer processes
        filter_fn = lambda args: '--type=renderer' in args
        
        client_id_to_pid = {}
        def extract_renderer_id(pid, args):
            # Extract the renderer client ID from the process arguments
            match = re.search(r'--renderer-client-id=(\d+)', ' '.join(args))
            if match:
                client_id_to_pid[int(match.group(1))] = pid            
                
        walk_child_processes(self.driver.service.process.pid, filter_fn, extract_renderer_id)
        return client_id_to_pid


    def kill_new_renderer_processes(self, prev_client_id_to_pid: Dict[int, int]) -> bool:
        is_renderer_processes_killed = False
        try:
            client_id_to_pid = self.collect_renderer_ids()
            
            # find new renderer processes
            new_client_id_to_pid = {client_id: pid for client_id, pid in client_id_to_pid.items() if client_id not in prev_client_id_to_pid}
            if len(new_client_id_to_pid) > 0:
                logger.info(f'Found new chrome renderer processes: {new_client_id_to_pid}')
            
                # Kill the process associated with the highest renderer client ID
                for client_id, pid in new_client_id_to_pid.items():
                    kill_process(pid)
                    logger.info(f'Killed chrome renderer process with client ID: {client_id}')
                is_renderer_processes_killed = True
            else:
                logger.info(f'No new chrome renderer processes found')
                
        except Exception as exc:
            logger.error(f'Error while killing the last renderer process: {str(exc)}')
            
        return is_renderer_processes_killed


    def driver_safe_close_current_tab(self, task_id: int, prev_client_id_to_pid: Optional[Dict[int, int]]) -> bool:
        """
        Close the current tab in the browser.
        Tries to close the tab gracefully, if it fails, kills new chrome renderer processes.
        
        Args:
            prev_client_id_to_pid: dict of renderer client IDs to process IDs before the tab was opened.
        
        Returns:
            True if the tab was closed successfully, False otherwise.
        """
        is_closed: bool = False
        close_timeout_secs = 8 # chrome tab gracefull close timeout in secs
        
        def close_driver_tab():
            try:
                self.driver.close()
            except Exception as exc:
                logger.error(f"[{task_id}] failed to safely close the chrome tab. {str(exc)}")
                pass

        # try first to close tab gracefully
        close_tab_thread = threading.Thread(target=close_driver_tab)
        close_tab_thread.start()
        close_tab_thread.join(timeout=close_timeout_secs)
        
        if close_tab_thread.is_alive():
            # close tab thread is still alive
            logger.error(f'[{task_id}] Chrome tab close timeout ({close_timeout_secs} secs) exceeded. Killing new chrome renderer processes...')
        else:
            is_closed = True
            
        # kill new renderer processes to stop any pending operations
        # after that it should successfully close the tab
        if not is_closed:
            try:
                if self.kill_new_renderer_processes(prev_client_id_to_pid):
                    # close current tab - it may hang after new renderer processes were killed
                    close_tab_thread = threading.Thread(target=close_driver_tab)
                    close_tab_thread.start()
                    close_tab_thread.join(timeout=close_timeout_secs / 2)
                    
                    if close_tab_thread.is_alive():
                        logger.error(f'[{task_id}] after killing new renderer processes still failed to close chrome tab in {close_timeout_secs / 2} secs')
                        self.reopen(task_id)
                    else:
                        is_closed = True
                        logger.info(f'[{task_id}] closed chrome tab after killing new renderer processes')
            except Exception as exc:
                logger.error(f"[{task_id}] failed to safely close the chrome tab. {str(exc)}")
                pass
        
        return is_closed


    def open_new_tab(self, task_id: int) -> bool:
        try:
            self.driver.switch_to.new_window('tab')
            logger.info(f' [{task_id}] opened new Chrome tab...')
            return True
        except Exception as exc:
            logger.error(f' [{task_id}] error opening new Chrome tab. {str(exc)}')
            return False


    VALIDATE_URL_LOAD_TIME: bool = False        

    def get_additional_metadata_for_url(self, get_task_id: int,
                                        input_url: str, input_title_string: str, timeout_in_secs: float,
                                        max_results_to_collect: int = MAX_SEARCH_RESULTS):
        corresponding_description_string = ''
        list_of_date_strings_fixed = []
        list_of_base64_encoded_images = []
        list_of_corresponsing_image_urls = []
        is_page_can_be_loaded = asyncio.run(validate_url_load_time(input_url, timeout_in_secs / 2)) if self.VALIDATE_URL_LOAD_TIME else True
            
        is_new_tab_opened = False
        if is_page_can_be_loaded:
            current_window_handle = self.driver.current_window_handle
            client_id_to_pid = None
            try:
                query = f"site:{input_url}"
                encoded_query_url = f"https://www.google.com/search?q={quote(query)}"
                client_id_to_pid = self.collect_renderer_ids()
                is_new_tab_opened = self.open_new_tab(get_task_id)
                if not is_new_tab_opened:
                    return corresponding_description_string, list_of_date_strings_fixed, list_of_base64_encoded_images, list_of_corresponsing_image_urls
                logger.info(f' [{get_task_id}] getting search metadata for [{input_url}]...')
                google_search_page_html = self.driver_retrieve_url_data(get_task_id, encoded_query_url, METADATA_URL_LOAD_TIMEOUT_SECS)
                corresponding_description_string = extract_corresponding_description_string_from_input_title_string_func(input_title_string, google_search_page_html)
                logger.info(f' [{get_task_id}] getting metadata for [{input_url}]...')
                time.sleep(random.uniform(1.5, 2.5))
                underlying_url_html = self.driver_retrieve_url_data(get_task_id, input_url, METADATA_URL_LOAD_TIMEOUT_SECS)
                time.sleep(random.uniform(1.5, 2.5))
                if self.driver_safe_close_current_tab(get_task_id, client_id_to_pid):
                    self.driver.switch_to.window(current_window_handle)    
                is_new_tab_opened = False
                if corresponding_description_string:
                    list_of_date_strings_fixed = extract_valid_dates_from_string_func(corresponding_description_string)
                else:
                    list_of_date_strings_fixed = []
                list_of_base64_encoded_images, list_of_corresponsing_image_urls = \
                        asyncio.run(get_all_images_on_page_as_base64_encoded_strings(get_task_id, underlying_url_html, min_image_size_to_retrieve=METADATA_MIN_IMAGE_SIZE_TO_RETRIEVE_BYTES,
                                                                         timeout_in_secs=timeout_in_secs, max_results_to_collect=max_results_to_collect))
            except Exception as e:
                logger.info(f' [{get_task_id}] could not retrieve additional metadata for [{input_url}]. {str(e)}')
                if is_new_tab_opened:
                    if self.driver_safe_close_current_tab(get_task_id, client_id_to_pid):
                        self.driver.switch_to.window(current_window_handle)
        else:
            logger.info(f' [{get_task_id}] could not retrieve additional metadata for [{input_url}]. Page load time may exceed {timeout_in_secs} seconds.')
        return corresponding_description_string, list_of_date_strings_fixed, list_of_base64_encoded_images, list_of_corresponsing_image_urls


    def get_results_of_reverse_image_search(self):
        current_graph_json = None
        status_result__search = self.search_google_image_search_for_image()
        if status_result__search:
            logger.info(f'\nImage search result found')
        else:
            logger.info(f'\nThere was a problem with the reverse image search!')
        soup = BeautifulSoup(self.driver.page_source, "lxml")    
        logger.info('Done parsing reverse image search page!')

        div_elements = [x for x in soup.find_all('div', attrs={"data-action-url": True})
                        if (lambda elem_str: '.gstatic.com' in elem_str and len(elem_str) < 45000)(str(x))
                        and x.find('a', {"aria-label": True, "role": "link"})  # The div should contain an 'a' tag with an href attribute.
                        and "Search" not in x.get_text()  # The div shouldn't contain the text "Search".
                        ]
        min_number_of_exact_matches_in_page = len(div_elements)
        combined_summary_df = pd.DataFrame(
            columns=['title',
                     'description_text',
                     'original_url',
                     'date_string',
                     'resolution_string',
                     'img_src_string',
                     'misc_related_image_url',
                     'misc_related_image_as_b64_string'],
            index=pd.Index([], name='search_result_ranking'))
        
        timer = Timer(True)
        images_hashes = set()
        current_index = 0
        get_task_id: int = 0
        for div_index, current_result in enumerate(div_elements):
            title_string, primary_url, img_src, resolution = get_fields_from_image_search_result(current_result)
            # stop enumeration if we collected MAX_SEARCH_RESULTS images
            image_count = len(combined_summary_df)
            get_task_id += 1
            if image_count < MAX_SEARCH_RESULTS:
                logger.info(f'Collected {image_count} images. Getting additional metadata [{div_index + 1}/{len(div_elements)}] for URL [{primary_url}]'
                            f' (max items to collect: {MAX_SEARCH_RESULTS - image_count})...')
                description, list_of_date_strings_fixed, list_of_base64_encoded_images, list_of_corresponsing_image_urls = \
                    self.get_additional_metadata_for_url(get_task_id, primary_url, title_string,
                                                         METADATA_DOWNLOAD_TIMEOUT_SECS, MAX_SEARCH_RESULTS - image_count)
            else:
                logger.info(f"Collected total {image_count} images in {timer.elapsed_time:.3f} secs")
                break
            date_string = "|".join(list_of_date_strings_fixed)
            # keep only images with correct base64 encoding and remove duplicates
            need_to_save_img_src = True
            img_src_resampled = resample_img_src(img_src, 1000)
            for base64_encoded_image_data, image_url in zip(list_of_base64_encoded_images, list_of_corresponsing_image_urls):
                _, base64_data_only = extract_base64_data_from_image_url(base64_encoded_image_data)
                if not base64_data_only:
                    continue
                image_base64_data_hash = get_sha3_256_from_data(base64_data_only)
                if image_base64_data_hash in images_hashes:
                    continue
                images_hashes.add(image_base64_data_hash)
                # add one dataframe row for each retrieved image
                new_row = pd.DataFrame(
                    [[
                        title_string,
                        description,
                        primary_url,
                        date_string,
                        resolution,
                        img_src_resampled if need_to_save_img_src else '',
                        image_url,
                        base64_data_only
                    ]], columns=combined_summary_df.columns, index=[current_index])
                current_index += 1
                need_to_save_img_src = False
                combined_summary_df = pd.concat([combined_summary_df, new_row], ignore_index=False)
        del div_elements
        gc.collect()

        if len(combined_summary_df) > 0:
            combined_list_of_base64_encoded_images = combined_summary_df['misc_related_image_as_b64_string'].values.tolist()
            search_images = [ImageDataPreProcessedBase64String(base64_encoded_data) for base64_encoded_data in combined_list_of_base64_encoded_images]
            indices_to_keep, _, rare_on_internet__similarity_df, rare_on_internet__adjacency_df = \
                filter_out_dissimilar_images_batch('google image search results', self.resized_image_save_path, search_images, 
                                                   self.dd_params.is_task_cancelled, CONFIG.image_processing_batch_size)
            self.check_task_cancelled()
            if len(indices_to_keep) < MAX_RESULTS_TO_RETURN:
                logger.info(f'Keeping {len(indices_to_keep)} of {len(search_images)} '
                            'google reverse image search images that are above the similarity score threshold.')
            else:
                logger.info(f'Keeping {MAX_RESULTS_TO_RETURN} of {len(search_images)} '
                            'google reverse image search images that are above the similarity score threshold '
                           f'(truncated from {len(indices_to_keep)} images).')
                indices_to_keep = indices_to_keep[:MAX_RESULTS_TO_RETURN]
                rare_on_internet__similarity_df = rare_on_internet__similarity_df.head(MAX_RESULTS_TO_RETURN)
                rare_on_internet__adjacency_df = rare_on_internet__adjacency_df.head(MAX_RESULTS_TO_RETURN)
                
            filtered_combined_summary_df = combined_summary_df.loc[indices_to_keep]
            # reset index to be sequential 
            filtered_combined_summary_df.reset_index(drop=True, inplace=True)
            # replace image data with resampled from search_images
            for i, index_to_keep in enumerate(indices_to_keep):
                filtered_combined_summary_df.loc[i, 'misc_related_image_as_b64_string'] = search_images[index_to_keep].get_base64_encoded_data()
            del combined_summary_df
            gc.collect()
            current_graph_json = generate_rare_on_internet_graph_func(filtered_combined_summary_df, rare_on_internet__similarity_df, rare_on_internet__adjacency_df)
        else:
            filtered_combined_summary_df = combined_summary_df
            logger.warning('\n\n\n************WARNING: No valid images extracted!!************\n\n\n')
        return min_number_of_exact_matches_in_page, filtered_combined_summary_df, current_graph_json


    def get_ip_func(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP


    def prepare_image_for_serving_func(self):
        sha3_256_hash_of_image_file = get_hash_sha3_256_from_file(self.resized_image_save_path)
        image_format = self.resized_image_save_path.suffix[1:]
        destination_path = CONFIG.served_files_path / sha3_256_hash_of_image_file[0:10] + '.' + image_format[0:3]
        shutil.copy(self.resized_image_save_path, destination_path)
        logger.info(f'Copied file {str(self.resized_image_save_path)} to path {str(destination_path)}')
        destination_url = 'http://' + self.get_ip_func() + '/' + sha3_256_hash_of_image_file[0:10] + '.' + image_format
        cleanup_old_files(CONFIG.served_files_path, "thumbnail image", None, None, 3)
        return destination_path, destination_url


    def is_jquery_loaded(self):
        """
        Check if the page is using Ajax (jQuery) to load content
        
        Returns:
            bool: True if jQuery is loaded, False otherwise
        """
        try:
            is_loaded = self.driver.execute_script("return typeof jQuery != 'undefined'")
        except Exception as exc:
            is_loaded = False
        return is_loaded
    

    def extract_data_from_google_lens_page_func(self):
        list_of_images_as_base64__filtered = []
        list_of_img_src_strings__filtered = []
        list_of_alt_strings__filtered = []
        list_of_href_strings__filtered = []
        alt_list_of_image_base64_hashes_filtered = []
        try:
            logger.info('Attempting to retrieve Google Lens results for image...')
            _, _, a_elements_count_before_search = \
                self.search_google_lens_for_image_func()
            timer = Timer(True)
            a_elements_count: int = None
            id: int = 1
            while timer.elapsed_time < GOOGLE_LENS_RESULT_PAGE_TIMEOUT:
                sleep_time = random.uniform(0.5, 1.0)
                logger.info(f' [{id}] sleep for {sleep_time:.3f} secs')
                time.sleep(sleep_time)
                self.wait_for_page_loaded(id, self.img.file_name)
                page_source = self.driver.page_source
                logger.info(f' [{id}] ...page is loaded ({len(page_source)} bytes)!')
                logger.info(f' [{id}] parsing page with BeautifulSoup...')
                soup = BeautifulSoup(page_source, "lxml")    
                logger.info(f' [{id}] done parsing page!')
                a_elements = [x for x in soup.find_all('a') if x.has_attr('aria-label') ]
                a_elements_count = len(a_elements)
                logger.info(f' [{id}] found {a_elements_count} <a> elements on the page')
                if a_elements_count > a_elements_count_before_search:
                    break
                id += 1
            a_elements_filtered = [x for x in a_elements if ' data-thumbnail-url=' in str(x) and ' data-item-title="' in str(x)]
            logger.info(f'Filtered {len(a_elements_filtered)} <a> elements with thumbnails on the page')
            a_elements_filtered_strings = [str(x) for x in a_elements_filtered]
            logger.info(f'Extracted {len(a_elements_filtered_strings)} strings')
            list_of_alt_strings = [(x.split(' data-item-title="'))[1].split(' data-thumbnail-url=')[0] for x in a_elements_filtered_strings]
            logger.info(f'Found {len(list_of_alt_strings)} alt strings')
            list_of_img_src_strings = [(x.split(' data-thumbnail-url='))[1].split(' jsaction="')[0] for x in a_elements_filtered_strings]
            list_of_img_src_strings = [x.replace('"','') for x in list_of_img_src_strings] #get rid of quote marks before and after the url
            logger.info(f'Found {len(list_of_img_src_strings)} img_src strings')
            list_of_href_strings = [(x.split('" href="'))[1].split('" role="')[0] for x in a_elements_filtered_strings]
            logger.info(f'Found {len(list_of_href_strings)} href strings')
            logger.info('Getting images and storing as base64 strings...')
            if len(list_of_img_src_strings) > MAX_SEARCH_RESULTS:
                logger.info(f"Truncating image search results [{len(list_of_img_src_strings)}] -> [{MAX_SEARCH_RESULTS}]")
                list_of_alt_strings = list_of_alt_strings[:MAX_SEARCH_RESULTS]                    
                list_of_img_src_strings = list_of_img_src_strings[:MAX_SEARCH_RESULTS]                    
                list_of_href_strings = list_of_href_strings[:MAX_SEARCH_RESULTS]
            current_graph_json = ''
            list_of_images_as_base64__filtered = []
            alt_list_of_image_base64_hashes_filtered = []
            if len(list_of_img_src_strings) > 0:
                try:
                    list_of_images = [ImageDataPreProcessedUrl(image_src_url) for image_src_url in list_of_img_src_strings]
                    list_of_image_indices_to_keep, alt_list_of_image_base64_hashes_filtered, \
                    alt_rare_on_internet__similarity_df, alt_rare_on_internet__adjacency_df = \
                        filter_out_dissimilar_images_batch('google lens results', self.resized_image_save_path, list_of_images,
                                                           self.dd_params.is_task_cancelled, CONFIG.image_processing_batch_size)
                    self.check_task_cancelled()
                    logger.info(f'Keeping {len(list_of_image_indices_to_keep)} of {len(list_of_images)} google lens images that are above the similarity score threshold.')
                    if len(list_of_image_indices_to_keep) > 0:
                        list_of_img_src_strings__filtered = np.array(list_of_img_src_strings)[list_of_image_indices_to_keep].tolist()
                        list_of_images_as_base64__filtered = [list_of_images[index].get_thumbnail() for index in list_of_image_indices_to_keep]
                        list_of_alt_strings__filtered = list(np.array(list_of_alt_strings)[list_of_image_indices_to_keep])
                        list_of_href_strings__filtered = list(np.array(list_of_href_strings)[list_of_image_indices_to_keep])
                        alt_rare_on_internet__similarity_df = alt_rare_on_internet__similarity_df.loc[alt_list_of_image_base64_hashes_filtered, alt_list_of_image_base64_hashes_filtered]
                        alt_rare_on_internet__adjacency_df = alt_rare_on_internet__adjacency_df.loc[alt_list_of_image_base64_hashes_filtered, alt_list_of_image_base64_hashes_filtered]
                    else:
                        logger.info('Zero images kept in google lens results! Either image was not found, or there was some kind of problem!')

                    current_graph_json = generate_alt_rare_on_internet_graph_func(list_of_images_as_base64__filtered, 
                                                                                  list_of_img_src_strings__filtered,
                                                                                  list_of_alt_strings__filtered,
                                                                                  list_of_href_strings__filtered,
                                                                                  alt_list_of_image_base64_hashes_filtered,
                                                                                  alt_rare_on_internet__similarity_df,
                                                                                  alt_rare_on_internet__adjacency_df)
                except BaseException as e:
                    logger.exception('Encountered problem filtering out dissimilar images from Google Lens data')

            alternative_rare_on_internet_graph_json_compressed_b64 = compress_text_data_with_zstd_and_encode_as_base64(current_graph_json)

            dict_of_google_lens_results = {'list_of_image_src_strings': list_of_img_src_strings__filtered,
                                           'list_of_image_alt_strings': list_of_alt_strings__filtered,
                                           'list_of_images_as_base64': list_of_images_as_base64__filtered,
                                           'list_of_sha3_256_hashes_of_images_as_base64': alt_list_of_image_base64_hashes_filtered,
                                           'list_of_href_strings': list_of_href_strings__filtered,
                                           'alternative_rare_on_internet_graph_json_compressed_b64': alternative_rare_on_internet_graph_json_compressed_b64}
            dict_of_google_lens_results_as_json = json.dumps(dict_of_google_lens_results, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            return dict_of_google_lens_results_as_json
        except BaseException as e:
            logger.exception('Problem getting Google Lens data')
            dict_of_google_lens_results_as_json = ''
            return dict_of_google_lens_results_as_json


    def click_button_until_disappears(self, name: str, xpath: str, max_attempts: int = 10) -> bool:
        logger.info(f"Attempting to click '{name}' button...")
        for attempt in range(max_attempts):
            try:
                wait = WebDriverWait(self.driver, 10) # wait up to 10 seconds
                button_element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
                
                # Scroll the button into view
                self.driver.execute_script("arguments[0].scrollIntoView();", button_element)
                
                # Click the button using JavaScript
                self.driver.execute_script("arguments[0].click();", button_element)
                
                # Wait for page to load after click
                wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')                
                
                # Create a BeautifulSoup object from the current page source
                current_soup = BeautifulSoup(self.driver.page_source, "lxml")
                
                # Check if the button text still exists in the page
                if current_soup.body(text=xpath):
                    logger.info(f"Button '{name}' text still exists after click, attempt {attempt + 1}")
                    time.sleep(random.uniform(0.5, 1.2))
                else:
                    logger.info(f"Button '{name}' text has disappeared, click successful!")
                    return True
            except (TimeoutException, NoSuchElementException):
                logger.info(f"Button '{name}' not found or not clickable, attempt {attempt + 1}")
                time.sleep(random.uniform(0.5, 1.2))
        logger.info(f"Failed to click the button '{name}' after {max_attempts} attempts")
        return False
    
    
    def search_google_image_search_for_image(self) -> bool:
        is_succeeded = False
        try:
            with time_limit(40 if not is_in_debugger() else DEBUG_TIME_LIMIT_SECS, 'Search google image search for image.'):
                try:
                    resized_image_save_path = CONFIG.resized_images_top_save_path / self.img.file_name
                    self.resized_image_save_path = resized_image_save_path
                    self.img.save_thumbnail(resized_image_save_path)
                    google_reverse_image_search_base_url = 'https://www.google.com/imghp?hl=en'
                    self.driver.get(google_reverse_image_search_base_url)
                    time.sleep(random.uniform(1.2, 2.5))
                    list_of_annoying_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'No thanks')]")
                    if len(list_of_annoying_buttons) > 0:
                        for current_button in list_of_annoying_buttons:
                            try:
                                actions = ActionChains(self.driver)
                                actions.move_to_element(current_button)
                                actions.perform()
                                current_button.click()
                                logger.info('Had to click -No Thanks- button when asked to login to a Google account. Done successfully.')
                            except BaseException as e:
                                logger.exception('Could not click on the -No Thanks- button')
                    try:
                        buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'I agree')]")
                        if len(buttons) > 0:
                            for current_button in buttons:
                                try:
                                    actions = ActionChains(self.driver)
                                    actions.move_to_element(current_button)
                                    actions.perform()
                                    current_button.click()
                                    logger.info('Had to click -I Agree- button. Done successfully.')
                                except BaseException as e:
                                    logger.exception('Could not click on the -I Agree- button')
                    except:
                        pass
                    try:
                        buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Accept all')]")
                        if len(buttons) > 0:
                            for current_button in buttons:
                                try:
                                    actions = ActionChains(self.driver)
                                    actions.move_to_element(current_button)
                                    actions.perform()
                                    current_button.click()
                                    logger.info('Had to click -Accept all- button. Done successfully.')
                                except BaseException as e:
                                    logger.exception('Could not click on the -Accept all- button')
                    except:
                        pass
                    logger.info('Trying to select "Search by Image" button...')
                    wait = WebDriverWait(self.driver, 10) # wait up to 10 seconds
                    search_by_image_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[aria-label='Search by image']")))
                    actions = ActionChains(self.driver)
                    if search_by_image_button:
                        logger.info('Found "Search by Image" button, now trying to click it...')
                        actions.move_to_element(search_by_image_button).perform()
                        search_by_image_button.click()
                        logger.info('Clicked "Search by Image" button!')
                    time.sleep(random.uniform(1, 2))

                    try:
                        file_selector = self.driver.find_element(By.XPATH, "//input[@type='file' and @name='encoded_image']")
                        if not file_selector:
                            raise Exception('Could not find file selector control!')
                        logger.info('Found file selector control, now trying to send file path to it...')
                        self.driver.execute_script("arguments[0].style.display = 'block';", file_selector)
                        file_selector.send_keys(str(resized_image_save_path))
                        time.sleep(random.uniform(0.2, 0.4))
                        logger.info('...sent file path to file selector control!')
                    except:                    
                        logger.info('Trying to click the "upload a file" button...')
                        button_xpath = "//span[@role='button' and contains(text(), 'upload a file')]"
                        list_of_buttons = self.driver.find_elements(By.XPATH, button_xpath)
                        EC.element_to_be_clickable((By.XPATH, button_xpath))
                        if len(list_of_buttons) > 0:
                            logger.info(f'Found "upload an image" button [{len(list_of_buttons)}], now trying to click it...')
                            for current_button in list_of_buttons:
                                try:
                                    actions = ActionChains(self.driver)
                                    actions.move_to_element(current_button)
                                    actions.perform()
                                    current_button.click()
                                    logger.info('Clicked "upload an image" button!')
                                    time.sleep(random.uniform(0.2, 0.4))
                                    logger.info(f'Sending the following file path string to file upload selector control:\n{str(resized_image_save_path)}\n')
                                    file_selector = self.driver.find_element(By.XPATH, "//input[@type='file']")
                                    self.driver.execute_script("arguments[0].style.display = 'block';", file_selector)
                                    time.sleep(random.uniform(0.2, 0.4))
                                    file_selector.send_keys(str(resized_image_save_path))
                                    logger.info('Sent file path to file selector control!')

                                    # Send the Escape key to close the dialog
                                    file_selector.send_keys(SeleniumKeys.ESCAPE)
                                    logger.info('Sent Escape key to close the dialog!')                                
                                except:
                                    pass
                    time.sleep(random.uniform(2.5, 5))
                   
                    def find_element_safe(by, value):
                        try:
                            return self.driver.find_element(by, value)
                        except NoSuchElementException:
                            return None
                        
                    def adjust_slider(slider_name: str, slider_handle, desired_value: int, direction_multiplier: int, is_horizontal: bool):
                        if not slider_handle:
                            return
                        
                        max_iterations = 30
                        # Get the initial value of the slider
                        initial_value = int(slider_handle.get_attribute("value"))
                        new_value = initial_value
                        
                        # Only adjust if necessary
                        if (direction_multiplier == 1 and initial_value < desired_value) or (direction_multiplier == -1 and initial_value > desired_value):
                            logger.info(f"Adjusting {slider_name} slider from {initial_value} to {desired_value}")
                            offset = 10 # initial offset
                            total_offset = 0 # total offset applied
                            change = 0
                            iterations = 0 # safety check - counter for number of iterations
                            
                            while change == 0 and iterations < max_iterations:
                                try:                            
                                    # Try a small move to see how much the value changes
                                    actions.move_to_element(slider_handle).click_and_hold().move_by_offset(
                                        offset * direction_multiplier if is_horizontal else 0,
                                        -offset * direction_multiplier if not is_horizontal else 0).release().perform()
                            
                                    # Get the changed value
                                    new_value = int(slider_handle.get_attribute("value"))
                                    change = new_value - initial_value
                                    total_offset += offset
                                    
                                    if change == 0: # If no change, increase the move offset
                                        offset += 5
                                    else:
                                        logger.info(f"{slider_name} slider value changed to {new_value}")
                                except MoveTargetOutOfBoundsException:
                                    # If the move is out of bounds, reduce the offset and try again
                                    offset -= 5
                                    offset = max(offset, 1)
                                except BaseException:
                                    # If anything else goes wrong, stop trying
                                    break
                                iterations += 1
                            
                            if change != 0 and new_value != desired_value:
                                # Calculate the required move to set the value to the desired_value
                                total_move = direction_multiplier * (desired_value - new_value) * (total_offset / change)
                                try:
                                    actions.move_to_element(slider_handle).click_and_hold().move_by_offset(
                                        total_move if is_horizontal else 0,
                                        -total_move if not is_horizontal else 0).release().perform()
                                except:
                                    pass
                                
                                # Check value after the move and adjust if necessary
                                final_value = int(slider_handle.get_attribute("value"))
                                logger.info(f"{slider_name} slider final value is {final_value}, offset {total_offset}, change {change}")
                                while (final_value != desired_value) and (iterations < max_iterations):
                                    try:
                                        if direction_multiplier == 1:
                                            adjustment_offset = total_offset * direction_multiplier if final_value < desired_value else 0
                                        else:
                                            adjustment_offset = total_offset * direction_multiplier if final_value > desired_value else 0
                                        if adjustment_offset == 0:
                                            break
                                        actions.move_to_element(slider_handle).click_and_hold().move_by_offset(
                                            adjustment_offset if is_horizontal else 0,
                                            -adjustment_offset if not is_horizontal else 0).release().perform()
                                        new_final_value = int(slider_handle.get_attribute("value"))
                                        if final_value != new_final_value:
                                            logger.info(f"{slider_name} slider final value adjusted to {new_final_value}")
                                            final_value = new_final_value
                                    except:
                                        pass
                                    iterations += 1                                

                    try:
                        top_left_slider = find_element_safe(By.XPATH, "//input[@type='range' and starts-with(@aria-label, 'top left corner')]")
                        top_right_slider = find_element_safe(By.XPATH, "//input[@type='range' and starts-with(@aria-label, 'top right corner')]")
                        bottom_right_slider = find_element_safe(By.XPATH, "//input[@type='range' and starts-with(@aria-label, 'bottom right corner')]")
                        bottom_left_slider = find_element_safe(By.XPATH, "//input[@type='range' and starts-with(@aria-label, 'bottom left corner')]")
                        
                        if top_left_slider and bottom_right_slider and top_right_slider and bottom_left_slider:
                            adjust_slider("top left", top_left_slider, 0, -1, True)  # move top left corner to the left
                            adjust_slider("bottom right", bottom_right_slider, 100, 1, True)  # move bottom right corner to the right
                            adjust_slider("bottom left", bottom_left_slider, 0, -1, False)  # move bottom left corner to the bottom
                            adjust_slider("top right", top_right_slider, 100, 1, False) # move top right corner to the top
                    except:
                        pass
                    
                    if not self.click_button_until_disappears("Find image source", "//button[.//div[text()='Find image source']]"):
                        raise Exception('Could not click on the "Find image source" button')
                    time.sleep(random.uniform(2.0, 3.0))
                    is_succeeded = True
                except BaseException as exc:
                    logger.exception('Problem using Selenium driver, now trying with local HTTP server')
                    try:
                        destination_path, destination_url = self.prepare_image_for_serving_func()
                        logger.info('Local http link for image: ' + destination_url)
                        img_check = DDImage(destination_path)
                        if img_check:
                            logger.info(f'File {destination_path} is a valid image')
                            hosted_image_url_response_code = urllib.request.urlopen(destination_url).getcode()
                            if hosted_image_url_response_code == httpx.codes.OK:
                                google_reverse_image_search_base_url = 'https://www.google.com/searchbyimage?q=&image_url=' + destination_url
                                self.driver.get(google_reverse_image_search_base_url)
                                is_succeeded = True
                            else:
                                logger.error('Unable to access served image!')
                                raise ValueError('Unable to access served image for reverse image search')
                        else:
                            logger.error('Resized image is not a valid image!')
                            raise ValueError('Resized image for reverse image search is invalid!')
                    except BaseException as e:
                        logger.exception('Problem using local HTTP hosting, now trying with Imgur upload!')
                        uploaded_image = im.upload_image(str(resized_image_save_path),
                                                        title="PastelNetwork: " + str(datetime.now()))
                        time.sleep(random.uniform(2.5, 5.5))
                        logger.info('Imgur link: ' + uploaded_image.link)
                        google_reverse_image_search_base_url = 'https://www.google.com/searchbyimage?q=&image_url=' + uploaded_image.link
                        self.driver.get(google_reverse_image_search_base_url)
                        is_succeeded = True
        except BaseException:
            logger.exception('Encountered Error with "Rare on Internet" check')
        return is_succeeded


    def search_google_lens_for_image_func(self) -> Tuple[int, str, int]:
        status_result: int = 0
        a_elements_count: int = 0
        with time_limit(40 if not is_in_debugger() else DEBUG_TIME_LIMIT_SECS, 'Search Google Lens for image.'):
            try:
                resized_image_save_path = CONFIG.resized_images_top_save_path / self.img.file_name
                self.resized_image_save_path = resized_image_save_path
                self.img.save_thumbnail(resized_image_save_path)
                google_lens_base_url = 'https://lens.google.com/search?p='
                self.driver.get(google_lens_base_url)
                time.sleep(random.uniform(1.2, 2.5))
                list_of_annoying_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'No thanks')]")
                if len(list_of_annoying_buttons) > 0:
                    for current_button in list_of_annoying_buttons:
                        try:
                            actions = ActionChains(self.driver)
                            actions.move_to_element(current_button)
                            actions.perform()
                            current_button.click()
                            logger.info('Had to click -No Thanks- button when asked to login to a Google account. Done successfully.')
                        except BaseException as e:
                            logger.info('Could not click on the -No Thanks- button. Error message: ' + str(e))
                try:
                    buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'I agree')]")
                    if len(buttons) > 0:
                        buttons[0].click()
                        logger.info('Had to click -I agree- box. Done successfully.')
                except:
                    pass
                try:
                    buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Accept all')]")
                    if len(buttons) > 0:
                        for current_button in buttons:
                            try:
                                actions = ActionChains(self.driver)
                                actions.move_to_element(current_button)
                                actions.perform()
                                current_button.click()
                                logger.info('Had to click -Accept all- button. Done successfully.')
                            except BaseException as e:
                                logger.info('Could not click on the -Accept all- button. Error message: ' + str(e))
                except:
                    pass
                time.sleep(random.uniform(0.3, 0.6))
                list_of_annoying_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'No thanks')]")
                if len(list_of_annoying_buttons) > 0:
                    for current_button in list_of_annoying_buttons:
                        try:
                            actions = ActionChains(self.driver)
                            actions.move_to_element(current_button)
                            actions.click(on_element=current_button)
                            actions.perform()
                            logger.info('Had to click -No Thanks- button when asked to login to a Google account. Done successfully.')
                        except BaseException as e:
                            logger.info('Could not click on the -No Thanks- button. Error message: ' + str(e))
                time.sleep(random.uniform(0.5, 3))
                
                
                logger.info('Now trying to click the Upload button...')
                list_of_buttons = self.driver.find_elements(By.XPATH, "//span[contains(text(), 'Upload')]")
                if len(list_of_buttons) > 0:
                    for current_button in list_of_buttons:
                        try:
                            actions = ActionChains(self.driver)
                            actions.move_to_element(current_button)
                            actions.click(on_element=current_button)
                            actions.perform()
                            logger.info('Successfully clicked Upload button!')
                            time.sleep(random.uniform(0.1, 0.2))
                        except:
                            pass
                        
                logger.info('Now trying to click the Computer button...')
                list_of_buttons = self.driver.find_elements(By.XPATH, "//span[contains(text(), 'Computer')]")
                if len(list_of_buttons) > 0:
                    for current_button in list_of_buttons:
                        try:
                            actions = ActionChains(self.driver)
                            actions.move_to_element(current_button)
                            actions.click(on_element=current_button)
                            actions.perform()
                            logger.info('Successfully clicked Computer button!')
                            time.sleep(random.uniform(0.1, 0.2))
                        except:
                            pass
                time.sleep(random.uniform(0.5, 1.0))
                choose_file_button_element = self.driver.find_element(By.XPATH, "//input[@type='file']")
                if choose_file_button_element:
                    logger.info('Found file input element')
                self.driver.execute_script("arguments[0].style.display = 'block';", choose_file_button_element)
                # count "a" elements before we send image path
                logger.info('Parsing original search page with BeautifulSoup...')
                soup = BeautifulSoup(self.driver.page_source, "lxml")    
                logger.info('Done parsing page!')
                a_elements = [x for x in soup.find_all('a') if x.has_attr('aria-label') ]
                a_elements_count = len(a_elements)
                logger.info(f'Found {a_elements_count} <a> elements on the page')
                
                choose_file_button_element.send_keys(str(resized_image_save_path))
                status_result = 1
            except BaseException as e:
                logger.exception('Problem getting Google lens data')
        return status_result, resized_image_save_path, a_elements_count


    def get_list_of_similar_images_func(self):
        status_result = 0
        list_of_urls_of_images_in_page = list()
        list_of_urls_of_visually_similar_images = list()
        try:
            url_elements_on_page = self.driver.find_elements(By.XPATH, "//a[@href]")
            if len(url_elements_on_page) > 0:
                for indx, elem in enumerate(url_elements_on_page):
                    current_url = elem.get_attribute("href")
                    if "imgres?imgurl" in current_url:
                        list_of_urls_of_images_in_page = list_of_urls_of_images_in_page + [current_url]
                    if "tbs=simg:" in current_url:
                        list_of_urls_of_visually_similar_images = list_of_urls_of_visually_similar_images + [
                            current_url]
                list_of_urls_of_images_in_page__clean = [x.split('imgurl=')[-1].split('&imgrefurl')[0] for x in
                                                         list_of_urls_of_images_in_page]
                list_of_urls_of_images_in_page__clean = remove_dupes_from_list_but_preserve_order_func(list_of_urls_of_images_in_page__clean)
                list_of_urls_of_visually_similar_images = remove_dupes_from_list_but_preserve_order_func(list_of_urls_of_visually_similar_images)
                status_result = 1
        except BaseException as e:
            logger.exception('Encountered Error')
            list_of_urls_of_images_in_page__clean = list()
        return status_result, list_of_urls_of_images_in_page__clean, list_of_urls_of_visually_similar_images


    def get_number_of_pages_of_search_results_func(self):
        number_of_pages_of_results = 1
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            initial_next_page_button_element = self.driver.find_elements(By.CSS_SELECTOR, "[aria-label='Next page']")
            if len(initial_next_page_button_element)> 0:
                initial_next_page_button_element[0].click()
            list_of_search_results_start_strings = list()
            url_elements_on_page = self.driver.find_elements(By.XPATH, "//a[@href]")
            for indx, elem in enumerate(url_elements_on_page):
                current_url = elem.get_attribute("href")
                if ('&start=' in current_url) and ('google.com/search?' in current_url):
                    current_search_start_string = current_url.split('&start=')[-1].split('&')[0]
                    list_of_search_results_start_strings = list_of_search_results_start_strings + [
                        current_search_start_string]
            list_of_search_results_start_strings = remove_dupes_from_list_but_preserve_order_func(list_of_search_results_start_strings)
            number_of_pages_of_results__method_1 = max([1, len(list_of_search_results_start_strings)])
            pager_elements_on_page = self.driver.find_elements(By.XPATH, "//a[contains(@aria-label, 'Page ')]")
            number_of_pages_of_results__method_2 = len(pager_elements_on_page) + 1
            pager_elements_on_page_alternative = self.driver.find_elements(By.XPATH,
                                                                           "//span[contains(@style,'background:url(/images/nav')]")
            number_of_pages_of_results__method_3 = len(pager_elements_on_page_alternative) + 1
            number_of_pages_of_results = max(
                [3, number_of_pages_of_results__method_1, number_of_pages_of_results__method_2,
                 number_of_pages_of_results__method_3])
            logger.info('Counted ' + str(
                number_of_pages_of_results) + ' pages of Google search results for reverse image query!')
            self.driver.back()

        except BaseException as e:
            logger.exception('Encountered Error getting number of pages of search results')
        return number_of_pages_of_results

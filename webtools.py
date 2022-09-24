# Copyright (c) 2021-2021 The Pastel Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
import io
import os, time, socket, hashlib, shutil, urllib.request, re, random, base64, json, pathlib

import chromedriver_autoinstaller
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

from lib.utils import DDImage, MyTimer
from lib.logger import logger
from lib.config import Config
from datetime import datetime
import datefinder
import requests
from html.parser import HTMLParser
from html import unescape
import pyimgur
from fuzzywuzzy import process

import torch
from lib import models
from lib.models.gem import GeneralizedMeanPoolingP
import torchvision
import torchvision.transforms
from torch import nn
from torch.utils.data import Dataset
import faiss
import h5py
import numpy as np
from selenium.common.exceptions import NoSuchElementException
import re
import zstandard as zstd
from bs4 import BeautifulSoup

import traceback
from contextlib import contextmanager
import threading
import _thread

CLIENT_ID = "689300e61c28cc7"
CLIENT_SECRET = "6c45e31ca3201a2d8ee6709d99b76d249615a10c"
im = pyimgur.Imgur(CLIENT_ID, CLIENT_SECRET)

SERVED_FILES_PATH = os.path.expanduser('~/pastel_dupe_detection_service/img_server/')


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


def get_sha256_hash_of_input_data_func(input_data_or_string):
    if isinstance(input_data_or_string, str):
        input_data_or_string = input_data_or_string.encode('utf-8')
    sha256_hash_of_input_data = hashlib.sha3_256(input_data_or_string).hexdigest()
    return sha256_hash_of_input_data


def get_image_hash_from_image_file_path_func(path_to_art_image_file):
    try:
        with open(path_to_art_image_file, 'rb') as f:
            art_image_file_binary_data = f.read()
        sha256_hash_of_art_image_file = get_sha256_hash_of_input_data_func(art_image_file_binary_data)
        return sha256_hash_of_art_image_file
    except Exception as e:
        logger.error('Error: ' + str(e))


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


def generate_rare_on_internet_graph_func(combined_summary_df, rare_on_internet__similarity_df, rare_on_internet__adjacency_df, old_code):
    links_list = []
    nodes_list = []
    list_of_result_titles = combined_summary_df['title'].values.tolist()
    list_of_search_result_rankings = [x+1 for x in combined_summary_df['search_result_ranking'].values.tolist()]
    list_of_img_src_strings = combined_summary_df['img_src_string'].values.tolist()
    list_of_img_alt_strings = combined_summary_df['img_alt_string'].values.tolist()
    list_of_image_resolutions = combined_summary_df['resolution_string'].values.tolist()
    list_of_link_urls_google_cache = combined_summary_df['google_cached_url'].values.tolist()
    list_of_link_urls_primary = combined_summary_df['original_url'].values.tolist()
    list_of_dates = combined_summary_df['date_string'].values.tolist()
    list_of_image_labels = rare_on_internet__adjacency_df.index.tolist()
    image_label_to_result_title_dict = dict(zip(list_of_image_labels, list_of_result_titles))
    image_label_to_search_result_ranking_dict = dict(zip(list_of_image_labels, list_of_search_result_rankings))
    image_label_to_img_src_dict = dict(zip(list_of_image_labels, list_of_img_src_strings))
    image_label_to_img_alt_dict = dict(zip(list_of_image_labels, list_of_img_alt_strings))
    image_label_to_image_resolutions_dict = dict(zip(list_of_image_labels, list_of_image_resolutions))
    image_label_to_link_urls_primary_dict = dict(zip(list_of_image_labels, list_of_link_urls_primary))
    image_label_to_link_urls_google_cache_dict = dict(zip(list_of_image_labels, list_of_link_urls_google_cache))
    image_label_to_dates_dict = dict(zip(list_of_image_labels, list_of_dates))
    cntr = 0
    for ii, label1 in enumerate(list_of_image_labels):
        try:
            nodes_list = nodes_list + [dict(id=ii,
                                        image_label=label1,
                                        title=image_label_to_result_title_dict[label1],
                                        search_result_ranking=image_label_to_search_result_ranking_dict[label1],
                                        img_src_string=image_label_to_img_src_dict[label1],
                                        img_alt_string=image_label_to_img_alt_dict[label1],
                                        resolution_string=image_label_to_image_resolutions_dict[label1],
                                        original_url=image_label_to_link_urls_primary_dict[label1],
                                        google_cached_url=image_label_to_link_urls_google_cache_dict[label1],
                                        date_string=image_label_to_dates_dict[label1],
                                        )]
        except BaseException as e:
            logger.error('Encountered error adding node to rare on internet graph structure: ' + str(e))
        for jj, label2 in enumerate(list_of_image_labels):
            current_adjacency = rare_on_internet__adjacency_df.iloc[ii, jj]
            if current_adjacency and label1 != label2:
                try:
                    cntr = cntr + 1
                    links_list = links_list + [dict(source=ii, target=jj, connection_strength=str(rare_on_internet__similarity_df.loc[label1, label2]))]
                except BaseException as e:
                    logger.error('Encountered error adding link to rare on internet graph structure: ' + str(e))     
    current_graph = {'nodes': nodes_list, 'links': links_list}
    #print('current_graph', current_graph)
    current_graph_json = json.dumps(current_graph)
    return current_graph_json


def generate_alt_rare_on_internet_graph_func(list_of_images_as_base64__filtered, list_of_image_src_strings__filtered, list_of_image_alt_strings__filtered, alt_list_of_image_base64_hashes_filtered, alt_rare_on_internet__similarity_df, alt_rare_on_internet__adjacency_df):
    links_list = []
    nodes_list = []
    list_of_images_as_base64 = list_of_images_as_base64__filtered
    list_of_image_hashes = alt_list_of_image_base64_hashes_filtered
    list_of_img_src_strings = list_of_image_src_strings__filtered
    list_of_img_alt_strings = list_of_image_alt_strings__filtered
    list_of_image_labels =  alt_rare_on_internet__adjacency_df.index.tolist()
    image_label_to_image_base64_string = dict(zip(list_of_image_labels, list_of_images_as_base64))
    image_label_to_image_hash_dict = dict(zip(list_of_image_labels, list_of_image_hashes))
    image_label_to_img_src_dict = dict(zip(list_of_image_labels, list_of_img_src_strings))
    image_label_to_img_alt_dict = dict(zip(list_of_image_labels, list_of_img_alt_strings))
    cntr = 0
    for ii, label1 in enumerate(list_of_image_labels):
        try:
            nodes_list = nodes_list + [dict(id=ii,
                                            image_label=label1,
                                            image_base64_string = image_label_to_image_base64_string[label1],
                                            sha3_256_hash_of_image_base64_string=image_label_to_image_hash_dict[label1],
                                            img_src=image_label_to_img_src_dict[label1],
                                            img_alt=image_label_to_img_alt_dict[label1],
                                            )]
        except BaseException as e:
            logger.error('Encountered error adding node to alternative rare on internet graph structure: ' + str(e))
        for jj, label2 in enumerate(list_of_image_labels):
            current_adjacency = alt_rare_on_internet__adjacency_df.iloc[ii, jj]
            if current_adjacency and label1 != label2:
                try:
                    cntr = cntr + 1
                    links_list = links_list + [dict(source=ii, target=jj, connection_strength=str(alt_rare_on_internet__similarity_df.loc[label1, label2]))]
                except BaseException as e:
                    logger.error('Encountered error adding link to alternative rare on internet graph structure: ' + str(e))                    
    current_graph = {'nodes': nodes_list, 'links': links_list}
    current_graph_json = json.dumps(current_graph)
    return current_graph_json

def compress_text_data_with_zstd_and_encode_as_base64_func(input_text_data):
    zstd_compression_level = 22  # Highest (best) compression level is 22
    zstandard_compressor = zstd.ZstdCompressor(level=zstd_compression_level, write_content_size=True, write_checksum=True)
    input_text_data_compressed = zstandard_compressor.compress(input_text_data.encode('utf-8'))
    input_text_data_compressed_b64 = base64.b64encode(input_text_data_compressed).decode('utf-8')  
    return input_text_data_compressed_b64


class ChromeDriver:

    def __init__(self, config: Config, img: DDImage):
        chromedriver_autoinstaller.install()
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1200")
        chrome_options.add_argument("--disable-gpu")
        # chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_experimental_option("useAutomationExtension", False)
        prefs = {"download.default_directory": config.internet_rareness_downloaded_images_path}
        chrome_options.add_experimental_option("prefs", prefs)
        self.resized_image_save_path = None
        self.driver = webdriver.Chrome(options=chrome_options)  # executable_path=chromedriver_path,
        # self.driver.implicitly_wait(3)
        self.config = config
        self.img = img
        
    def filter_out_dissimilar_images_func(self, list_of_images_as_base64):
        logger.info("Now filtering out dissimilar images from google image results...")
        model = models.create("resnet50", num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        checkpoint_file = self.config.support_files_path + 'DupeDetector_gray.pth.tar'
        print("Loading checkpoint... ", checkpoint_file)
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        model = model.cpu()
        if torch.cuda.is_available():
            pass
            #model.cuda()

        def preprocess_image(img_path):
            mean, std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
            transforms = [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean, std)
                ]
            transforms = torchvision.transforms.Compose(transforms)
            img = Image.open(img_path)
            img = img.convert("RGB")
            img = torchvision.transforms.RandomGrayscale(p=1)(img)
            img = img.resize((256,256))
            img = transforms(img).reshape((1,3,256,256))
            if torch.cuda.is_available():
                pass
                #img = img.cuda()
            return img
        
        def base64_to_preimage(base64_str):
            from PIL import Image
            import base64
            import io
            imgdata = base64.b64decode(base64_str)
            return io.BytesIO(imgdata)
        
        def read_descriptors(filenames):
            """ read descriptors from a set of HDF5 files """
            descs = []
            names = []
            for filename in filenames:
                print("Loading...", filename)
                hh = h5py.File(filename, "r")
                descs.append(np.array(hh["vectors"]))
                names += np.array(hh["image_names"][:], dtype=object).astype(str).tolist()
            # strip paths and extensions from the filenames
            names = [
                name.split('/')[-1]
                for name in names
            ]
            names = [
                name[:-4] if name.endswith(".jpg") or name.endswith(".png") else name
                for name in names
            ]
            return names, np.vstack(descs)
        
        all_desc_q = []
        features = model.module.base(preprocess_image(self.resized_image_save_path))
        features = features.view(features.size(0), -1)
        all_desc_q.append(features)
        all_desc_q = torch.vstack(tuple(all_desc_q)).cpu().detach().numpy()
        logger.info("Load PCA matrix q")
        pca = faiss.read_VectorTransform(self.config.support_files_path + 'pca_bw.vt')
        print(f"Apply PCA q {pca.d_in} -> {pca.d_out}")
        all_desc_q = pca.apply_py(all_desc_q)
        logger.info("normalizing descriptors q")
        faiss.normalize_L2(all_desc_q)
        all_desc_q.resize((all_desc_q.shape[1], all_desc_q.shape[0]))
        print("The shape of feature is q", all_desc_q.shape)
        print("all_desc_q", all_desc_q)
        all_desc_r = []
        for r_str in range(len(list_of_images_as_base64)):
            features = model.module.base(preprocess_image(base64_to_preimage(list_of_images_as_base64[r_str])))
            features = features.view(features.size(0), -1)
            all_desc_r.append(features)
        all_desc_r = torch.vstack(tuple(all_desc_r)).cpu().detach().numpy()
        logger.info("Load PCA matrix r")
        pca = faiss.read_VectorTransform(self.config.support_files_path + 'pca_bw.vt')
        print(f"Apply PCA q {pca.d_in} -> {pca.d_out}")
        all_desc_r = pca.apply_py(all_desc_r)
        logger.info("normalizing descriptors r")
        faiss.normalize_L2(all_desc_r)
        print("The shape of feature is r", all_desc_r.shape)
        print("all_desc_r", all_desc_r)
        cos_scores = (all_desc_r)@all_desc_q
        for iu in range(len(cos_scores)):
            print("The cos_scores are",cos_scores[iu])
        numpy_feature_of_query = all_desc_q.reshape(1, all_desc_q.shape[0])
        print("Now calculating similarity matrix of rare on the internet images...")
        rare_on_internet__cos_similarity_matrix = all_desc_r @ (all_desc_r.T)
        
        rare_on_internet__cos_similarity_matrix_df = pd.DataFrame(rare_on_internet__cos_similarity_matrix)
        rare_on_internet__cos_similarity_matrix_df = rare_on_internet__cos_similarity_matrix_df.round(8)
        train_image_ids, xt = read_descriptors([self.config.support_files_path + 'train_0_bw.hdf5'])
        index = faiss.IndexFlatIP(xt.shape[1])
        index.add(xt)
        train_scores, I = index.search(numpy_feature_of_query.astype('float32'), 10)
        score_norms = train_scores.mean(axis=1) * 2
        cos_scores = cos_scores - score_norms
        min_normalized_cos_similarity_threshold__lowest_permissible = -0.18
        target_pct_of_results_to_keep = 0.35
        print('Target percentage of results to keep: ', target_pct_of_results_to_keep)
        if len(list_of_images_as_base64) >= 3:
            min_normalized_cos_similarity_threshold = 0.0
            number_of_images_above_similarity_threshold = np.sum([(x > min_normalized_cos_similarity_threshold).astype(int) for idx, x in enumerate(cos_scores)])
            pct_of_images_above_similarity_threshold = number_of_images_above_similarity_threshold / len(cos_scores)
            while ( (pct_of_images_above_similarity_threshold < target_pct_of_results_to_keep) and (min_normalized_cos_similarity_threshold >= min_normalized_cos_similarity_threshold__lowest_permissible) ):
                # print('Current minimum normalized cosine similarity threshold: ', min_normalized_cos_similarity_threshold)
                # print('Current number of results above the min. threshold: ', number_of_images_above_similarity_threshold)
                # print('Current percentage of results above the min. threshold: ', pct_of_images_above_similarity_threshold)
                min_normalized_cos_similarity_threshold = min_normalized_cos_similarity_threshold - 0.001
                pct_of_images_above_similarity_threshold = np.sum([(x > min_normalized_cos_similarity_threshold).astype(int) for idx, x in enumerate(cos_scores)]) / len(cos_scores)
        else:
            min_normalized_cos_similarity_threshold = -0.25
            pct_of_images_above_similarity_threshold = 0
        print('Selected final threshold value of: ', min_normalized_cos_similarity_threshold)
        print('Final percentage of results above the min. threshold: ', pct_of_images_above_similarity_threshold)

        list_of_image_indices_to_keep = []
        for iu in range(len(cos_scores)):
            print("The cos_scores_norm are",cos_scores[iu])
            if cos_scores[iu] > min_normalized_cos_similarity_threshold:
                list_of_image_indices_to_keep.append(iu)
        print("The list_of_image_indices_to_keep are:", list_of_image_indices_to_keep)
        list_of_image_labels = ['Image_' + str(idx).zfill(2) for idx, x in enumerate(list_of_image_indices_to_keep)]
        rare_on_internet__similarity_df = rare_on_internet__cos_similarity_matrix_df.iloc[list_of_image_indices_to_keep, list_of_image_indices_to_keep]
        rare_on_internet__similarity_df.index = list_of_image_labels
        rare_on_internet__similarity_df.columns = list_of_image_labels
        list_of_image_base64_hashes_filtered = rare_on_internet__similarity_df.index.tolist()
        similarity_threshold_for_connection = 0.30
        rare_on_internet__adjacency_df = (rare_on_internet__similarity_df >= similarity_threshold_for_connection).astype(int)
        return list_of_image_indices_to_keep, list_of_image_base64_hashes_filtered, rare_on_internet__similarity_df, rare_on_internet__adjacency_df 
        

    def try_to_get_table_from_page(self):
        try:
            try: #Wait until there are at least three visible images in the search results page (not necessarily three actual results). This will be true because of the Visually similar images section even if there are no real results.
                WebDriverWait(self.driver, 10).until(lambda wd: len([el for el in wd.find_element(By.ID,"search").find_elements(By.XPATH,'.//img') if el.is_displayed()])>3)
            except Exception as e: 
                logger.error('Error encountered trying to find the "search" element on the page using the ID field...' + str(e))
            print('Now parsing page with BeautifulSoup...')
            soup = BeautifulSoup(self.driver.page_source, "lxml")    
            print('Done parsing page!')
            div_elements = [x for x in soup.find_all('div')]
            div_elements_filtered = [x for x in div_elements if x.has_attr('data-sokoban-container') ]
            div_elements_filtered2 = [x for x in div_elements_filtered if 'data:image' in str(x)]
            sokoban_elements = [list(x.children)[1] for x in div_elements_filtered2]
            sokoban_element_strings = [str(x) for x in sokoban_elements]
            img_tags = [str(x.select('img')) for x in sokoban_elements]
            list_of_img_src_strings = [x.split('src="')[-1].split(' ')[0].split(' ')[0].split('"/>')[0].strip().replace('"','') for x in img_tags]
            # print('list_of_img_src_strings: ', list_of_img_src_strings)
            list_of_img_alt_strings = [x.split('img alt="')[-1].split('" c')[0].strip() for x in img_tags]
            print('list_of_img_alt_strings: ', list_of_img_alt_strings)
            href_list = [x.split('<a href="')[-1].split('" ping')[0] for x in sokoban_element_strings]
            sokoban_elements_more = [list(x.children)[0] for x in div_elements_filtered2]
            sokoban_element_more_strings = [str(x) for x in sokoban_elements_more]
            list_of_resolution_strings = [x.split('×')[0].split('<span>')[-1] + '×' + x.split('×')[-1].split('</span>')[0] for x in sokoban_element_more_strings]
            print('list_of_resolution_strings: ', list_of_resolution_strings)
        except BaseException as e:
            logger.error('Encountered Error on the first pass of getting Rare on the Internet data using normal method, now trying again with other approach! Error: ' + str(e))
            page_source_text = self.driver.page_source
            problem_files_path = pathlib.Path(self.config.support_files_path + 'rare_on_the_internet_problem_pages/')
            problem_files_path.mkdir(parents=True, exist_ok=True)
            self.remove_old_rare_on_internet_diagnostic_files_func(problem_files_path)
            problem_page_output_path = str(problem_files_path) + '/main_code__problem_page__' + datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '.html'
            print('\nSaving problem page to disk for analysis: ' + problem_page_output_path)
            with open(problem_page_output_path, 'w') as f:
                f.write(page_source_text)
            status, current_page_results_df = self.try_to_get_table_from_page_old()
            return status, current_page_results_df
        try:
            list_of_date_strings = [list(set(list(datefinder.find_dates(x)))) for x in sokoban_element_more_strings]
            list_of_date_strings_fixed = [x[0].isoformat().split('T0')[0].replace(' ','_').replace(':','_') if len(x) > 0 else '' for x in list_of_date_strings]
        except BaseException as e:
            logger.error('Error encountered parsing dates in the rare on the internet results-- trying again a different way...' + str(e))
            page_source_text = self.driver.page_source
            problem_files_path = pathlib.Path(self.config.support_files_path + 'rare_on_the_internet_problem_pages/')
            problem_files_path.mkdir(parents=True, exist_ok=True)
            self.remove_old_rare_on_internet_diagnostic_files_func(problem_files_path)
            problem_page_output_path = str(problem_files_path) + '/main_code__problem_page__' + datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '.html'
            print('\nSaving problem page to disk for analysis: ' + problem_page_output_path)
            with open(problem_page_output_path, 'w') as f:
                f.write(page_source_text)
            list_of_date_strings_fixed = []
            for current_potential_date_string in sokoban_element_more_strings:
                current_date_string = ''
                current_date_string_fixed = ''
                try:
                    current_date_string = list(datefinder.find_dates(current_potential_date_string))
                except:
                    print('Could not parse date from string: ', current_potential_date_string)
                if len(current_date_string) > 0:
                    try:
                        if current_date_string[0].isoformat() <= datetime.now():
                            current_date_string_fixed = current_date_string[0].isoformat().split('T0')[0].replace(' ','_').replace(':','_')
                        else:
                            print('Date was from the future, so obviously invalid! Ignoring...')
                            current_date_string_fixed = ''
                    except:
                        print('Could not parse date from string: ', current_date_string)
                list_of_date_strings_fixed.append(current_date_string_fixed)
        try:
            print('list_of_date_strings_fixed: ', list_of_date_strings_fixed)
            list_of_original_urls = [x.split(' href="')[1].split('" ping="')[0].split('">')[0].strip() for x in sokoban_element_more_strings]
            print('list_of_original_urls: ', list_of_original_urls)
            list_of_cached_urls = [x.split(' href="')[-1].split('" ping="')[0].split('">')[0].strip() for x in sokoban_element_more_strings]
            # print('list_of_cached_urls: ', list_of_cached_urls)
            list_of_titles = [str(x.select('h3')).split('">')[-1].split('</h3>]')[0].strip() for x in sokoban_elements_more]
            print('list_of_titles: ', list_of_titles)
            list_of_descriptions = [x.split('</span>')[-2].split('<span>')[-1].replace('<em>','').replace('</em>','').strip() for x in sokoban_element_more_strings]
            # print('list_of_descriptions: ', list_of_descriptions)
            number_of_elements_that_should_be_in_each_list = len(list_of_img_src_strings)
            print('number_of_elements_that_should_be_in_each_list: ', number_of_elements_that_should_be_in_each_list)
            if (len(list_of_titles) == number_of_elements_that_should_be_in_each_list) and \
            (len(list_of_descriptions) == number_of_elements_that_should_be_in_each_list) and \
            (len(list_of_original_urls) == number_of_elements_that_should_be_in_each_list) and \
            (len(list_of_cached_urls) == number_of_elements_that_should_be_in_each_list) and \
            (len(list_of_date_strings_fixed) == number_of_elements_that_should_be_in_each_list) and \
            (len(list_of_resolution_strings) == number_of_elements_that_should_be_in_each_list) and \
            (len(list_of_img_alt_strings) == number_of_elements_that_should_be_in_each_list):
                current_page_results_df = pd.DataFrame([list_of_titles, list_of_descriptions, list_of_original_urls, list_of_cached_urls, list_of_date_strings_fixed, list_of_resolution_strings, list_of_img_alt_strings, list_of_img_src_strings]).T
            else:
                current_page_results_df = pd.DataFrame()
                print('\n\nError! Inconsistent number of elements in lists:\n')
                print('len(list_of_titles): ', len(list_of_titles))
                print('len(list_of_descriptions): ', len(list_of_descriptions))
                print('len(list_of_original_urls): ', len(list_of_original_urls))
                print('len(list_of_cached_urls): ', len(list_of_cached_urls))
                print('len(list_of_date_strings_fixed): ', len(list_of_date_strings_fixed))
                print('len(list_of_resolution_strings): ', len(list_of_resolution_strings))
                print('len(list_of_img_alt_strings): ', len(list_of_img_alt_strings))
                print('len(list_of_img_src_strings): ', len(list_of_img_src_strings))
            current_page_results_df.columns = ['title', 'description_text', 'original_url', 'google_cached_url', 'date_string', 'resolution_string', 'img_alt_string', 'img_src_string']
            current_page_results_df = current_page_results_df[current_page_results_df['google_cached_url']!='']
            status = 1
        except BaseException as e:
            logger.error('Encountered Error getting rare on the internet using normal method, trying again with other approach! Error: ' + str(e))
            status, current_page_results_df = self.try_to_get_table_from_page_old()
            return status, current_page_results_df    
        return status, current_page_results_df
            

    def try_to_get_table_from_page_old(self):
        list_of_lists = []
        try: #Wait until there are at least three visible images in the search results page (not necessarily three actual results). This will be true because of the Visually similar images section even if there are no real results.
            WebDriverWait(self.driver, 10).until(lambda wd: len([el for el in wd.find_element(By.ID,"search").find_elements(By.XPATH,'.//img') if el.is_displayed()])>3)
        except Exception as e: 
            logger.error('Error encountered trying to find the "search" element on the page using the ID field...' + str(e))
        try:
            search_element = self.driver.find_element(By.ID, 'search')
        except BaseException as e:
            logger.error('Error encountered trying to select the "search" element on the page using the ID field...' + str(e))
            page_source_text = self.driver.page_source
            problem_files_path = pathlib.Path(self.config.support_files_path + 'rare_on_the_internet_problem_pages/')
            problem_files_path.mkdir(parents=True, exist_ok=True)
            self.remove_old_rare_on_internet_diagnostic_files_func(problem_files_path)
            problem_page_output_path = str(problem_files_path) + '/fallback_code__problem_page__' + datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '.html'
            print('\nSaving problem page to disk for analysis: ' + problem_page_output_path)
            with open(problem_page_output_path, 'w') as f:
                f.write(page_source_text)
        #search element has an h1 containing "Search Results" text and a data div contianing the actual search results divs. this is what we want
        search_element_child = search_element.find_element(By.XPATH,'*')
        #sort out visually similar images
        im_divs = search_element_child.find_elements(By.XPATH,'div/div[not(contains(.//h3, "Visually similar images"))]')
        #sort out results divs that don't have any images
        for im_div in im_divs:
            try:
                result_divs = im_div.find_elements(By.XPATH,'div[.//img]')
                if len(result_divs) > 0:
                    for idx, result_div in enumerate(result_divs):
                        current_result_group_html = result_div.find_element(By.XPATH, "div").get_attribute('outerHTML')
                        print('Index: ', idx)
                        soup = BeautifulSoup(current_result_group_html, "html.parser")
                        title_element = soup.find('h3')
                        print('title: ', title_element.text)
                        soup_text = soup.get_text()
                        list_of_a_elements = soup.find_all('a')
                        list_of_href_elements = list(set([x.get('href') for x in list_of_a_elements if re.match("^http",x.get('href')) != None ]))
                        print("list of href elements: ",("\n\t").join(list_of_href_elements))
                        cached_url = ''
                        primary_url = ''
                        if(len(list_of_href_elements) < 1 or len(list_of_href_elements)>2):
                            print("************WARNING: strange length of href elements in results div, refusing to continue************")
                            pass
                        for elem in list_of_href_elements:
                            if 'webcache.googleusercontent' in elem:
                                cached_url = elem
                            else:
                                primary_url = elem
                        description = soup_text
                        if("Cached" in description):
                            textSplitter = "Cached"
                            if("CachedSimilar" in soup_text):
                                textSplitter = "CachedSimilar"
                            description = soup_text.split(textSplitter)[1]
                        print("Description: ",description)
                        resolution = ''
                        s = re.search("(^\d+ × \d+)",description) #will only appear at the start of the description
                        if s != None:
                            resolution = str(s.group())
                        print("resolution: ", resolution)
                        date = ''
                        if('—' in description):
                            date_string = list(set(list(datefinder.find_dates(description))))
                            if len(date_string) > 0:
                                date_string_fixed = date_string[0].isoformat().split('T0')[0].replace(' ','_').replace(':','_')
                            else:
                                date_string_fixed = ''
                        print("Date: ", date_string_fixed)
                        img = soup.find("img")
                        alt = img.find("alt")
                        print("img: ", img)
                        img_src = img["src"]
                        img_alt = img["alt"]
                        list_of_lists.append([title_element.text, description, primary_url, cached_url, date_string_fixed,  resolution, img_alt, img_src])
            except NoSuchElementException:
                print('Found no img elements on current results page!')
                pass
        df = pd.DataFrame(list_of_lists,columns=['title', 'description_text', 'original_url', 'google_cached_url', 'date_string', 'resolution_string', 'img_alt_string', 'img_src_string'])
        df = df[df.google_cached_url != '']
        if len(df) == 0:
            logger.info('Error encountered using fallback Rare on the Internet code...')
            page_source_text = self.driver.page_source
            problem_files_path = pathlib.Path(self.config.support_files_path + 'rare_on_the_internet_problem_pages/')
            problem_files_path.mkdir(parents=True, exist_ok=True)
            problem_page_output_path = str(problem_files_path) + '/main_code__problem_page__' + datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '.html'
            print('\nSaving problem page to disk for analysis: ' + problem_page_output_path)
            with open(problem_page_output_path, 'w') as f:
                f.write(page_source_text)
        status = 1
        return status, df


    def get_results_of_reverse_image_search_func(self, old_code: bool):
        status_result = 0
        min_number_of_pages_of_results_to_get = 3
        max_number_of_pages_of_results_to_get = 5
        list_of_summary_dfs = []
        status_result__search = self.search_google_image_search_for_image_func()
        if status_result__search:
            logger.info(f'\nImage search result found')
        else:
            logger.info(f'\nThere was a problem with the reverse image search!')
        try:
            number_of_pages_of_results_available = self.get_number_of_pages_of_search_results_func()
            if number_of_pages_of_results_available < min_number_of_pages_of_results_to_get:
                logger.info('Counted fewer than 3 pages of results, but trying for 3 pages anyway...')
                number_of_pages_of_results_to_get = min_number_of_pages_of_results_to_get
            else:
                logger.info('Counted ' + str(number_of_pages_of_results_available) + ' of available pages...')
                number_of_pages_of_results_to_get = min(
                    [max_number_of_pages_of_results_to_get, number_of_pages_of_results_available])
                logger.info('Attempting to get ' + str(number_of_pages_of_results_to_get) + ' pages of results...')
        except BaseException as e:
            logger.error('Encountered Error getting number of pages of search results: ' + str(e))

        for current_page in range(number_of_pages_of_results_to_get):
            number_of_tries_so_far = 0
            extraction_successful = 0
            logger.info('\n\nNow getting page ' + str(current_page + 1) + ' of ' + str(number_of_pages_of_results_to_get) + ' for reverse image search...\n')
            
            number_of_times_to_try_new_code_before_reverting_to_old_code = 5
            if not old_code:
                while (not extraction_successful) and (number_of_tries_so_far < number_of_times_to_try_new_code_before_reverting_to_old_code):
                    number_of_tries_so_far = number_of_tries_so_far + 1
                    with MyTimer():
                        extraction_successful, summary_df = self.try_to_get_table_from_page()
                        # if len(summary_df) == 0:
                        #     print('Rare on the internet table is empty... trying again using alternative approach:')
                        #     page_source_text = self.driver.page_source
                        #     problem_files_path = pathlib.Path(self.config.support_files_path + 'rare_on_the_internet_problem_pages/')
                        #     problem_files_path.mkdir(parents=True, exist_ok=True)
                        #     problem_page_output_path = str(problem_files_path) + '/main_code__problem_page__' + datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '.html'
                        #     print('\nSaving problem page to disk for analysis: ' + problem_page_output_path)
                        #     with open(problem_page_output_path, 'w') as f:
                        #         f.write(page_source_text)
                        #     extraction_successful, summary_df = self.try_to_get_table_from_page_old()
            else:
                while (not extraction_successful) and (number_of_tries_so_far < number_of_times_to_try_new_code_before_reverting_to_old_code):
                    number_of_tries_so_far = number_of_tries_so_far + 1
                    with MyTimer():
                        extraction_successful, summary_df = self.try_to_get_table_from_page_old()
            print('Current Rare on the Internet summary_df:', summary_df)                                                                                
            list_of_summary_dfs = list_of_summary_dfs + [summary_df]
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                try:
                    next_page_elements = self.driver.find_elements(By.XPATH, "//a[contains(@id,'next')]")
                    if len(next_page_elements) > 0:
                        next_page_elements[0].click()
                except:
                    try:
                        next_page_elements = self.driver.find_elements(By.ID, "pnnext")
                        if len(next_page_elements) > 0:
                            next_page_elements[0].click()
                    except:
                        logger.error('Encountered Error trying to click next page element')
                time.sleep(random.uniform(0.2, 0.6))
            except BaseException as e:
                logger.error('Encountered Error scrolling to bottom for next page element: ' + str(e))

        try:
            list_of_sub_tables = [x for x in list_of_summary_dfs if len(x) > 0]
            if len(list_of_sub_tables) > 0:
                combined_summary_df = pd.concat(list_of_sub_tables, axis=0)
                #print("combined_summary_df length before drop dupes: " + str(len(combined_summary_df)))
                #combined_summary_df = combined_summary_df.drop_duplicates(subset=['result_title', 'title_y_coordinate'])
                combined_summary_df = combined_summary_df.drop_duplicates(subset=['title'])
                #print("combined_summary_df length after drop dupe titles: " + str(len(combined_summary_df)))
                if 'img_src_string' in combined_summary_df.columns.tolist():
                    combined_summary_df = combined_summary_df.dropna(subset=['img_src_string'])
                    combined_summary_df = combined_summary_df[
                        combined_summary_df['img_src_string'].str.contains('data:image/')]
                else:
                    logger.info('No exact image matches found in page, so cannot drop duplicates!')
                #print("combined_summary_df length after drop dupe img srcs: " + str(len(combined_summary_df)))
            else:
                combined_summary_df = pd.DataFrame()
            combined_summary_df = combined_summary_df.reset_index(drop=True)
            combined_summary_df['search_result_ranking'] = combined_summary_df.index

            #print("List of google reverse image search images before base64 split: " + ' '.join(map(str,combined_summary_df['img_src'].values.tolist())))
            if 'img_src_string' in combined_summary_df.columns.tolist():
                list_of_google_reverse_image_search_images_as_base64 = [x.split('base64,')[-1] for x in combined_summary_df['img_src_string'].values.tolist()]
                #print("List of google reverse image search images as base64: " + ' '.join(map(str,list_of_google_reverse_image_search_images_as_base64)))
                if len(list_of_google_reverse_image_search_images_as_base64) > 0:
                    list_of_google_reverse_image_search_image_indices_to_keep, list_of_image_base64_hashes_filtered, rare_on_internet__similarity_df, rare_on_internet__adjacency_df = self.filter_out_dissimilar_images_func(list_of_google_reverse_image_search_images_as_base64)
                    combined_summary_df__filtered = combined_summary_df.iloc[list_of_google_reverse_image_search_image_indices_to_keep, :]
                    if combined_summary_df__filtered.shape[0] == len(list_of_google_reverse_image_search_image_indices_to_keep):
                        print('Keeping ' + str(len(list_of_google_reverse_image_search_image_indices_to_keep)) + ' of ' +
                            str(len(list_of_google_reverse_image_search_images_as_base64)) + 
                            ' google reverse image search images that are above the similarity score threshold.')
                        combined_summary_df = combined_summary_df__filtered
                        if not old_code:
                            current_graph_json = generate_rare_on_internet_graph_func(combined_summary_df, rare_on_internet__similarity_df, rare_on_internet__adjacency_df, False)
                        else:
                            current_graph_json = generate_rare_on_internet_graph_func(combined_summary_df, rare_on_internet__similarity_df, rare_on_internet__adjacency_df, True)
            else:
                combined_summary_df = pd.DataFrame()
                current_graph_json = ''
            status_result = 1
        except BaseException as e:
            logger.error('Encountered Error combining sub-tables into combined table: ' + str(e))
            combined_summary_df = pd.DataFrame()
            current_graph_json = ''
            #this would seem to indicate that there was an actual error in processing vs 0 results, so send a status result that causes us to try old code
            status_result = 2
        try:
            if len(combined_summary_df['img_src_string']) == 0:
                logger.error(
                    '\n\n\n************WARNING: No valid images extracted for overall table!!************\n\n\n')
        except:
            pass
        try:
            if 'img_src_string' in combined_summary_df.columns.tolist():
                min_number_of_exact_matches_in_page = len(
                    [x for x in combined_summary_df['img_src_string'] if "data:image" in x])
            else:
                min_number_of_exact_matches_in_page = 0
                logger.info('No exact image matches found in page!')
        except BaseException as e:
            logger.error('Encountered Error getting min number of exact matches in page: ' + str(e))
            min_number_of_exact_matches_in_page = 0
        return status_result, combined_summary_df, min_number_of_exact_matches_in_page, current_graph_json


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

    def remove_old_served_image_files_func(self):
        limit_days = 3
        threshold = time.time() - limit_days * 86400
        list_of_file_paths = os.listdir(SERVED_FILES_PATH)
        try:
            for current_file_path in list_of_file_paths:
                creation_time = os.stat(os.path.join(SERVED_FILES_PATH, current_file_path)).st_ctime
                if creation_time < threshold:
                    logger.info('Removing old thumbnail image file that is past the ' + str(limit_days) +
                                '-day age limit: ' + current_file_path)
                    os.remove(current_file_path)
        except:
            logger.error('Error removing old thumbnail images...')

    def remove_old_rare_on_internet_diagnostic_files_func(self, problem_files_path):
        limit_days = 3
        threshold = time.time() - limit_days * 86400
        list_of_file_paths = os.listdir(problem_files_path)
        try:
            for current_file_path in list_of_file_paths:
                creation_time = os.stat(os.path.join(problem_files_path, current_file_path)).st_ctime
                if creation_time < threshold:
                    logger.info('Removing old Rare on the Internet diagnostic html file that is past the ' + str(limit_days) +
                                '-day age limit: ' + current_file_path)
                    os.remove(current_file_path)
        except:
            logger.error('Error removing old Rare on the Internet diagnostic html file...')

    def prepare_image_for_serving_func(self):
        sha3_256_hash_of_image_file = get_image_hash_from_image_file_path_func(self.resized_image_save_path)
        image_format = self.resized_image_save_path.split('.')[-1]
        destination_path = SERVED_FILES_PATH + sha3_256_hash_of_image_file[0:10] + '.' + image_format[0:3]
        shutil.copy(self.resized_image_save_path, destination_path)
        logger.info('Copied file ' + self.resized_image_save_path + ' to path ' + destination_path)
        destination_url = 'http://' + self.get_ip_func() + '/' + sha3_256_hash_of_image_file[0:10] + '.' + image_format
        self.remove_old_served_image_files_func()
        return destination_path, destination_url

    def get_content_and_generate_thumbnail_func(self, image_src_string: str) -> str:
        src_img_bytes = requests.get(image_src_string).content
        img_byte_arr = io.BytesIO(src_img_bytes)
        img = Image.open(img_byte_arr)
        conv_img = img.convert('RGB')
        conv_img.thumbnail((200, 200), resample=Image.LANCZOS)
        thumb_byte_arr = io.BytesIO()
        # conv_img.save(thumb_byte_arr, format='JPEG', subsampling=0, quality=70)
        conv_img.save(thumb_byte_arr, format='webp', quality=1, method=6, )
        enc_bytes = base64.b64encode(thumb_byte_arr.getvalue())
        readable_thumbnail_str = enc_bytes.decode('utf-8')
        return readable_thumbnail_str

    def extract_data_from_google_lens_page_func(self):
        try:
            logger.info('Now attempting to retrieve Google Lens results for image...')
            status_result__google_lens, resized_image_save_path = self.search_google_lens_for_image_func()
            logger.info('Now waiting for elements to be visible...')
            WebDriverWait(self.driver, 15).until(lambda wd: wd.find_element(By.XPATH, "//div[contains(@aria-live,'polite')]"))
            logger.info('Last wait')
            time.sleep(random.uniform(2, 4))
            google_lens_data_container_element = self.driver.find_elements(By.XPATH, "//div[contains(@aria-live,'polite')]")
            list_of_text_strings = []
            list_of_image_src_strings = []
            list_of_image_alt_strings = []
            list_of_images_as_base64 = []
            if len(google_lens_data_container_element) > 0:
                list_of_inner_html = [x.get_attribute('innerHTML') for x in google_lens_data_container_element]
                # print("Inner HTML", list_of_inner_html)
                parser = MyHTMLParser()
                if len(list_of_inner_html) > 0:
                    parser.feed(list_of_inner_html[0])
                    list_of_text_strings = parser.data
                    # print("List of text strings: ",list_of_text_strings)
                    list_of_text_strings = [unescape(x) for x in list_of_text_strings]
                else:
                    print("length of inner html less than 0")
            else:
                print("google lens data container element length zero")
            list_of_text_strings_sorted_by_length = sorted(list_of_text_strings, key=len)
            list_of_text_strings_sorted_by_length.reverse()
            list_of_text_strings_sorted_by_length__filtered1 = [x.replace('&amp;', '&') for x in
                                                                list_of_text_strings_sorted_by_length if 15 < len(x) < 100]
            list_of_text_strings_sorted_by_length__filtered2 = [x for x in list_of_text_strings_sorted_by_length__filtered1
                                                                if not ('$' in x and len(x) < 30)]
            list_of_text_strings_sorted_by_length__filtered3 = [x for x in list_of_text_strings_sorted_by_length__filtered2
                                                                if not ('.com' in x and len(x) < 30)]
            list_of_text_strings_sorted_by_length__filtered4 = [x for x in list_of_text_strings_sorted_by_length__filtered3
                                                                if not ('find what you were looking for?' in x)]
            list_of_text_strings_sorted_by_length__filtered5 = [x for x in list_of_text_strings_sorted_by_length__filtered4
                                                                if not ('Did you find these results useful?' in x)]
            list_of_text_strings_sorted_by_length__filtered6 = [x for x in list_of_text_strings_sorted_by_length__filtered5
                                                                if not ('Pages that include matching images' in x)]
            list_of_text_strings_sorted_by_length__filtered7 = [x for x in list_of_text_strings_sorted_by_length__filtered6
                                                                if not (
                        'Check website for latest pricing and availability' in x)]
            list_of_text_strings_sorted_by_length__filtered8 = [x for x in list_of_text_strings_sorted_by_length__filtered7
                                                                if not ('Lens Results' in x)]
            list_of_text_strings_sorted_by_length__filtered9 = [x for x in list_of_text_strings_sorted_by_length__filtered8
                                                                if not ('Related Results' in x)]

            list_of_strings_sorted_by_length__filtered_final = list_of_text_strings_sorted_by_length__filtered9
            logger.info('Text strings in alternative rare on internet results BEFORE fuzzy matching:\n')
            logger.info(list_of_strings_sorted_by_length__filtered_final)
            list_of_all_df_rows = []
            for idx1, current_string in enumerate(list_of_strings_sorted_by_length__filtered_final):
                current_string_match_results = process.extract(current_string,
                                                               [x for x in list_of_strings_sorted_by_length__filtered_final
                                                                if x != current_string], limit=30)
                current_list_of_df_rows = []
                for idx2, current_match_result in enumerate(current_string_match_results):
                    current_df_row = [current_string, current_match_result[0], current_match_result[1]]
                    current_list_of_df_rows.append(current_df_row)
                list_of_all_df_rows = list_of_all_df_rows + current_list_of_df_rows
            try:
                fuzzy_match_df = pd.DataFrame(list_of_all_df_rows)
                fuzzy_match_df.columns = ['source_string', 'match_string', 'similarity_score']
                similarity_score_threshold1 = 55.0
                similarity_score_threshold2 = 88.0
                fuzzy_match_df__filtered = fuzzy_match_df[fuzzy_match_df['similarity_score'] > similarity_score_threshold1]
                fuzzy_match_df__filtered = fuzzy_match_df__filtered[fuzzy_match_df__filtered['similarity_score'] < similarity_score_threshold2]
                print('fuzzy_match_df__filtered', fuzzy_match_df__filtered)
                combined_filtered_strings = fuzzy_match_df__filtered['match_string'].values.tolist()
                final_filtered_list_of_strings = list(set(combined_filtered_strings))
                final_filtered_list_of_strings_sorted_by_length = sorted(final_filtered_list_of_strings, key=len)
                final_filtered_list_of_strings_sorted_by_length.reverse()
                logger.info('Text strings in alternative rare on internet results AFTER fuzzy matching:\n')
                logger.info(final_filtered_list_of_strings_sorted_by_length)
                print('list_of_strings_sorted_by_length__filtered_final', list_of_strings_sorted_by_length__filtered_final)
            except BaseException as e:
                logger.error('Encountered Error putting fuzzy matches in dataframe: ' + str(e))
                final_filtered_list_of_strings_sorted_by_length = list_of_strings_sorted_by_length__filtered_final
            image_elements_on_page = self.driver.find_elements(By.XPATH, "//img")
            if len(image_elements_on_page) > 0:
                list_of_image_src_strings = [x.get_attribute("src").replace('blob:http', 'http') for x in
                                             image_elements_on_page]
                list_of_image_alt_strings = [x.get_attribute("alt") for x in image_elements_on_page]
                # logger.info('List of image src strings:\n')
                # logger.info(list_of_image_src_strings)
                list_boolean_filter = [('images/icons/material' not in x) and ('com/images/branding' not in x) and (len(x) > 100) for x
                                       in list_of_image_src_strings]
                list_of_image_src_strings__filtered = [x for idx, x in enumerate(list_of_image_src_strings) if
                                                       list_boolean_filter[idx]]
                list_of_image_alt_strings__filtered = [x for idx, x in enumerate(list_of_image_alt_strings) if
                                                       list_boolean_filter[idx]]
                list_of_images_as_base64 = [self.get_content_and_generate_thumbnail_func(x) for idx, x in
                                            enumerate(list_of_image_src_strings) if list_boolean_filter[idx]]
            if len(list_of_images_as_base64) > 0:
                try:
                    list_of_image_indices_to_keep, alt_list_of_image_base64_hashes_filtered, alt_rare_on_internet__similarity_df, alt_rare_on_internet__adjacency_df = self.filter_out_dissimilar_images_func(list_of_images_as_base64)
                    print('Keeping ' + str(len(list_of_image_indices_to_keep)) + ' of ' +
                            str(len(list_of_images_as_base64)) + 
                            ' google lens images that are above the similarity score threshold.')
                    if(len(list_of_image_indices_to_keep)) > 0:
                        list_of_images_as_base64__filtered = list(np.array(list_of_images_as_base64)[list_of_image_indices_to_keep][1:]) #the [1:] skips over the "visually analyzed image", which isn't a real search result
                        list_of_image_src_strings__filtered = list(np.array(list_of_image_src_strings__filtered)[list_of_image_indices_to_keep][1:])
                        list_of_image_alt_strings__filtered = list(np.array(list_of_image_alt_strings__filtered)[list_of_image_indices_to_keep][1:])
                        alt_list_of_image_base64_hashes_filtered = alt_list_of_image_base64_hashes_filtered[1:]
                        alt_rare_on_internet__similarity_df = alt_rare_on_internet__similarity_df.loc[alt_list_of_image_base64_hashes_filtered, alt_list_of_image_base64_hashes_filtered]
                        alt_rare_on_internet__adjacency_df = alt_rare_on_internet__adjacency_df.loc[alt_list_of_image_base64_hashes_filtered, alt_list_of_image_base64_hashes_filtered]
                    else:
                        print('Now images kept in google lens results!')
                        list_of_images_as_base64__filtered = []
                        list_of_image_src_strings__filtered = []
                        list_of_image_alt_strings__filtered = []
                        alt_list_of_image_base64_hashes_filtered = []
                    current_graph_json = generate_alt_rare_on_internet_graph_func(list_of_images_as_base64__filtered, 
                                                                                  list_of_image_src_strings__filtered,
                                                                                  list_of_image_alt_strings__filtered, 
                                                                                  alt_list_of_image_base64_hashes_filtered,
                                                                                  alt_rare_on_internet__similarity_df,
                                                                                  alt_rare_on_internet__adjacency_df)
                except BaseException as e:
                    logger.error('Encountered problem filtering out dissimilar images from Google Lens data:' + str(e))
            else:
                list_of_images_as_base64__filtered = list_of_images_as_base64
                alt_list_of_image_base64_hashes_filtered = [get_sha256_hash_of_input_data_func(x) for x in list_of_images_as_base64]
                current_graph_json = ''

            alternative_rare_on_internet_graph_json_compressed_b64 = compress_text_data_with_zstd_and_encode_as_base64_func(current_graph_json)

            dict_of_google_lens_results = {'list_of_text_strings': final_filtered_list_of_strings_sorted_by_length,
                                           'list_of_image_src_strings': list_of_image_src_strings__filtered,
                                           'list_of_image_alt_strings': list_of_image_alt_strings__filtered,
                                           'list_of_images_as_base64': list_of_images_as_base64__filtered,
                                           'list_of_sha3_256_hashes_of_images_as_base64': alt_list_of_image_base64_hashes_filtered,
                                           'alternative_rare_on_internet_graph_json_compressed_b64': alternative_rare_on_internet_graph_json_compressed_b64}
            dict_of_google_lens_results_as_json = json.dumps(dict_of_google_lens_results)
            return dict_of_google_lens_results_as_json
        except BaseException as e:
            logger.error('Problem getting Google Lens data:' + str(e))
            dict_of_google_lens_results_as_json = ''
            return dict_of_google_lens_results_as_json


    def search_google_image_search_for_image_func(self):
        try:
            with time_limit(40, 'Search google image search for image.'):
                try:
                    resized_image_save_path = self.config.resized_images_top_save_path + self.img.file_name
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
                                logger.info('Could not click on the -No Thanks- button. Error message: ' + str(e))
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
                                    logger.info('Could not click on the -I Agree- button. Error message: ' + str(e))
                    except:
                        pass
                    logger.info('Trying to select "Search by Image" button...')
                    search_by_image_button = self.driver.find_elements(By.CSS_SELECTOR, "[aria-label='Search by image']")
                    actions = ActionChains(self.driver)
                    if len(search_by_image_button):
                        logger.info('Found "Search by Image" button, now trying to click it...')
                        actions.move_to_element(search_by_image_button[0])
                        # actions.click(on_element=search_by_image_button[0])
                        actions.perform()
                        search_by_image_button[0].click()
                        logger.info('Clicked "Search by Image" button!')
                    time.sleep(random.uniform(1, 2))
                    
                    logger.info('Trying to click the "upload an image" button...')
                    list_of_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'upload')]")
                    EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'upload')]"))
                    if len(list_of_buttons) > 0:
                        logger.info('Found "upload an image" button, now trying to click it...')
                        for current_button in list_of_buttons:
                            try:
                                actions = ActionChains(self.driver)
                                actions.move_to_element(current_button)
                                actions.perform()
                                current_button.click()
                                logger.info('Clicked "upload an image" button!')
                                time.sleep(random.uniform(0.2, 0.4))
                                logger.info('Sending the following file path string to file upload selector control: \n' + resized_image_save_path + '\n')
                                file_selector = self.driver.find_element_by_xpath("//input[@type='file']")
                                self.driver.execute_script("arguments[0].style.display = 'block';", file_selector)
                                file_selector.send_keys(resized_image_save_path)
                                logger.info('Sent file path to file selector control!')
                            except:
                                pass
                    time.sleep(random.uniform(0.9, 1.2))
                    logger.info('Now attempting to click "Find image source" button...')
                    find_image_source_buttons = self.driver.find_elements(By.XPATH, "//a[contains(@aria-label, 'Find image source')]")
                    for current_button in find_image_source_buttons:
                        try:
                            actions = ActionChains(self.driver)
                            actions.move_to_element(current_button)
                            actions.perform()
                            current_button.click()
                            time.sleep(random.uniform(0.5, 0.9))
                            logger.info('Clicked "Find image source" button!')
                            self.driver.switch_to.window(self.driver.window_handles[-1])
                            status_result = 1
                        except:
                            logger.info('Error message: ' + str(e))
                            pass
                except BaseException as e:
                    logger.error('Problem using Selenium driver, now trying with local HTTP server. Error encountered: ' + str(e))
                    try:
                        destination_path, destination_url = self.prepare_image_for_serving_func()
                        logger.info('Local http link for image: ' + destination_url)
                        img_check = DDImage(destination_path)
                        if img_check:
                            logger.info(f'File {destination_path} is a valid image')
                            hosted_image_url_response_code = urllib.request.urlopen(destination_url).getcode()
                            if hosted_image_url_response_code == 200:
                                google_reverse_image_search_base_url = 'https://www.google.com/searchbyimage?q=&image_url=' + destination_url
                                self.driver.get(google_reverse_image_search_base_url)
                                status_result = 1
                            else:
                                logger.error('Unable to access served image!')
                                raise ValueError('Unable to access served image for reverse image search')
                        else:
                            logger.error('Resized image is not a valid image!')
                            raise ValueError('Resized image for reverse image search is invalid!')
                    except BaseException as e:
                        logger.error('Problem using local HTTP hosting, now trying with Imgur upload! Error encountered: ' + str(e))
                        uploaded_image = im.upload_image(resized_image_save_path,
                                                        title="PastelNetwork: " + str(datetime.now()))
                        time.sleep(random.uniform(2.5, 5.5))
                        logger.info('Imgur link: ' + uploaded_image.link)
                        google_reverse_image_search_base_url = 'https://www.google.com/searchbyimage?q=&image_url=' + uploaded_image.link
                        self.driver.get(google_reverse_image_search_base_url)
                        status_result = 1
        except BaseException as e:
            logger.error('Encountered Error with "Rare on Internet" check: ' + str(e))
            status_result = 0
        return status_result

    def search_google_lens_for_image_func(self):
        with time_limit(40, 'Search google lens for image.'):
            try:
                resized_image_save_path = self.config.resized_images_top_save_path + self.img.file_name
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
                list_of_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'file_upload')]")
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
                list_of_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'laptop_chromebook')]")
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
                # list_of_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Upload')]")
                # EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Upload')]"))
                # if len(list_of_buttons) > 0:
                #     for current_button in list_of_buttons:
                #         try:
                #             actions = ActionChains(self.driver)
                #             actions.move_to_element(current_button)
                #             actions.click(on_element=current_button)
                #             actions.perform()
                #             time.sleep(random.uniform(0.1, 0.2))
                #         except:
                #             pass
                # list_of_buttons2 = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Computer')]")
                # if len(list_of_buttons2) > 0:
                #     for current_button in list_of_buttons2:
                #         try:
                #             actions = ActionChains(self.driver)
                #             actions.move_to_element(current_button)
                #             actions.click(on_element=current_button)
                #             actions.perform()
                #             time.sleep(random.uniform(0.1, 0.2))
                #         except:
                #             pass
                time.sleep(random.uniform(0.1, 0.2))
                choose_file_button_element = self.driver.find_element_by_xpath("//input[@type='file']")
                self.driver.execute_script("arguments[0].style.display = 'block';", choose_file_button_element)
                choose_file_button_element.send_keys(resized_image_save_path)
                status_result = 1
            except BaseException as e:
                logger.error('Problem getting Google lens data:' + str(e))
                status_result = 0
        return status_result, resized_image_save_path

    def get_list_of_similar_images_func(self):
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
                list_of_urls_of_images_in_page__clean = list(set(list_of_urls_of_images_in_page__clean))
                list_of_urls_of_visually_similar_images = list(set(list_of_urls_of_visually_similar_images))
                status_result = 1
        except BaseException as e:
            logger.error('Encountered Error: ' + str(e))
            status_result = 0
            list_of_urls_of_images_in_page__clean = list()
        return status_result, list_of_urls_of_images_in_page__clean, list_of_urls_of_visually_similar_images

    def get_number_of_pages_of_search_results_func(self):
        number_of_pages_of_results = 1
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            list_of_search_results_start_strings = list()
            url_elements_on_page = self.driver.find_elements(By.XPATH, "//a[@href]")
            for indx, elem in enumerate(url_elements_on_page):
                current_url = elem.get_attribute("href")
                if ('&start=' in current_url) and ('google.com/search?' in current_url):
                    current_search_start_string = current_url.split('&start=')[-1].split('&')[0]
                    list_of_search_results_start_strings = list_of_search_results_start_strings + [
                        current_search_start_string]
            list_of_search_results_start_strings = list(set(list_of_search_results_start_strings))
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
        except BaseException as e:
            logger.error('Encountered Error getting number of pages of search results: ' + str(e))
        return number_of_pages_of_results

# *_*coding:utf-8 *_*
import os
import sys

############ For LINUX ##############
DATA_DIR = {
	'MER2024': 		 '/home/user/datasets/Emotion/MER2024',
    'CAER': 		 '/home/user/datasets/Emotion/CAER',
    'CMU-MOSEI':	 '/home/user/datasets/Emotion/CMU-MOSEI',
	'CMU-MOSI':      '/home/user/datasets/Emotion/CMU-MOSI',
	'DFEW': 		 '/home/user/datasets/Emotion/DFEW',
    'CH-SIMS_v2_s':  '/home/user/datasets/Emotion/CH-SIMS_v2_s',
    'MAFW': 		 '/home/user/datasets/Emotion/MAFW',
	'MELD': 		 '/home/user/datasets/Emotion/MELD',
	'IEMOCAP': 		 '/home/user/datasets/Emotion/IEMOCAP',
	'BOLD': 		 '/home/user/datasets/Emotion/BOLD',
	'E3': 		 	 '/home/user/datasets/Emotion/E3',
	'MC-EIU': 		 '/home/user/datasets/Emotion/MC-EIU',
	'MER2023-SEMI':	 '/home/user/datasets/Emotion/MER2023-SEMI',
	'SIMS':			 '/home/user/datasets/Emotion/SIMS',
	'OV-MERD+':		 '/home/user/datasets/Emotion/OV-MERD+',
}
PATH_TO_RAW_VIDEO = {
	'MER2024': os.path.join(DATA_DIR['MER2024'], 'video'),
	'CAER': os.path.join(DATA_DIR['CAER'], 'video'),
    'CMU-MOSEI': os.path.join(DATA_DIR['CMU-MOSEI'], 'video'),
    'CMU-MOSI': os.path.join(DATA_DIR['CMU-MOSI'], 'video'),
	'DFEW': os.path.join(DATA_DIR['DFEW'], 'video'),
    'CH-SIMS_v2_s': os.path.join(DATA_DIR['CH-SIMS_v2_s'], 'video'),
    'MAFW': os.path.join(DATA_DIR['MAFW'], 'video'),
	'MELD': os.path.join(DATA_DIR['MELD'], 'video'),
	'IEMOCAP': os.path.join(DATA_DIR['IEMOCAP'], 'video'),
	'BOLD': os.path.join(DATA_DIR['BOLD'], 'video'),
	'E3': os.path.join(DATA_DIR['E3'], 'video'),
	'MC-EIU': os.path.join(DATA_DIR['MC-EIU'], 'video'),
	'MER2023-SEMI': os.path.join(DATA_DIR['MER2023-SEMI'], 'video'),  
	'SIMS': os.path.join(DATA_DIR['SIMS'], 'video'),
 	'OV-MERD+': os.path.join(DATA_DIR['OV-MERD+'], 'video'),
}
PATH_TO_RAW_AUDIO = {
	'MER2024': os.path.join(DATA_DIR['MER2024'], 'audio'),
    'CAER': os.path.join(DATA_DIR['CAER'], 'audio'),
    'CMU-MOSEI': os.path.join(DATA_DIR['CMU-MOSEI'], 'audio'),
	'CMU-MOSI': os.path.join(DATA_DIR['CMU-MOSI'], 'audio'),
	'DFEW': os.path.join(DATA_DIR['DFEW'], 'audio'),
    'CH-SIMS_v2_s': os.path.join(DATA_DIR['CH-SIMS_v2_s'], 'audio'),
	'MAFW': os.path.join(DATA_DIR['MAFW'], 'audio'),
	'MELD': os.path.join(DATA_DIR['MELD'], 'audio'),
	'IEMOCAP': os.path.join(DATA_DIR['IEMOCAP'], 'audio'),
	'BOLD': os.path.join(DATA_DIR['BOLD'], 'audio'),
	'E3': os.path.join(DATA_DIR['E3'], 'audio'),
	'MC-EIU': os.path.join(DATA_DIR['MC-EIU'], 'audio'),
	'MER2023-SEMI': os.path.join(DATA_DIR['MER2023-SEMI'], 'audio'),
	'SIMS': os.path.join(DATA_DIR['SIMS'], 'audio'),
	'OV-MERD+': os.path.join(DATA_DIR['OV-MERD+'], 'audio'),
}

PATH_TO_FEATURES = {
	'MER2024': os.path.join(DATA_DIR['MER2024'], 'features'),
    'CAER': os.path.join(DATA_DIR['CAER'], 'features'),
    'CMU-MOSEI': os.path.join(DATA_DIR['CMU-MOSEI'], 'features'),
	'CMU-MOSI': os.path.join(DATA_DIR['CMU-MOSI'], 'features'),
	'DFEW': os.path.join(DATA_DIR['DFEW'], 'features'),
    'CH-SIMS_v2_s': os.path.join(DATA_DIR['CH-SIMS_v2_s'], 'features'),
	'MAFW': os.path.join(DATA_DIR['MAFW'], 'features'),
	'MELD': os.path.join(DATA_DIR['MELD'], 'features'),
	'IEMOCAP': os.path.join(DATA_DIR['IEMOCAP'], 'features'),
	'BOLD': os.path.join(DATA_DIR['BOLD'], 'features'),
	'E3': os.path.join(DATA_DIR['E3'], 'features'),
	'MC-EIU': os.path.join(DATA_DIR['MC-EIU'], 'features'),
	'MER2023-SEMI': os.path.join(DATA_DIR['MER2023-SEMI'], 'features'),
	'SIMS': os.path.join(DATA_DIR['SIMS'], 'features'),
	'OV-MERD+': os.path.join(DATA_DIR['OV-MERD+'], 'features'),
}



PATH_TO_FFMPEG = '/usr/bin/ffmpeg'
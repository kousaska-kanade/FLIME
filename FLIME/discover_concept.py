"""This script runs the whole ACE method.
for each class, extract the concept and store them in corresponding folder
"""
import argparse
import datetime
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from tcav import utils

import ace_helpers

import keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# ImageNet label for each class
class_dict = {
    # 'fight-plane':'n041455115',
    # 'battleship':'n041455116',
    # 'tank': 'n02123456',

    # 'Pistol':'n041455113',
    # 'assaultrifle':'n041455114',
     #'helicopter': 'n041455117',
    # 'howitzer': 'n041455120',
    #'passenger-plane': 'n041455119',
    #   "amphibious-assault-ship": "n041455121",
    #"sniper-rifle": "n041455123",
    # "main-battle-tank": "n041455122",
    # "rocket": "n041455124",
     'J-series-fighter':'n041455118',
}



def main(args):
    # for each class, extract the concept and store them in corresponding folder
    for target_class, filder in class_dict.items():
        print('start discovering concept of {}'.format(target_class))
        random_concept = 'random_discovery'  # Random concept for statistical testing
        sess = utils.create_session()


        if args.model_to_run != 'Resnet152':
            mymodel = ace_helpers.make_model(
                sess, args.model_to_run, args.model_path, args.labels_path)
        else:
            if args.gradcam_layer:
                mymodel = ace_helpers.make_model(
                    None, args.model_to_run, None, args.labels_path, gradcam_layer=args.gradcam_layer)
            else:
                mymodel = ace_helpers.make_model(
                    None, args.model_to_run, None, args.labels_path)

        ###### related DIRs on CNS to store results #######
        '''we change the folder to show the detected feature form grad-cam'''
        if args.bottlenecks:
            bottlenecks = args.bottlenecks.split(',')
            #print(bottlenecks)
            print(1111111)
        else:
            #print(bottlenecks)
            print(22222)
            bottlenecks = mymodel.find_target_layer()
            print(bottlenecks)

        discovered_concepts_dir = os.path.join(args.working_dir, 'output', target_class, 'concepts/')
        results_dir = os.path.join(args.working_dir, 'output', target_class, 'results/')
        cavs_dir = os.path.join(args.working_dir, 'output', target_class, 'cavs/')
        activations_dir = os.path.join(args.working_dir, 'output', target_class, 'acts/')
        results_summaries_dir = os.path.join(args.working_dir, 'output', target_class, 'results_summaries/')

        if not os.path.exists(discovered_concepts_dir):
            os.makedirs(discovered_concepts_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(cavs_dir):
            os.makedirs(cavs_dir)
        if not os.path.exists(activations_dir):
            os.makedirs(activations_dir)
        if not os.path.exists(results_summaries_dir):
            os.makedirs(results_summaries_dir)

        # Creating the ConceptDiscovery class instance
        cd = ConceptDiscovery(
            mymodel,
            # args.target_class,
            target_class,  # we run a for loop
            random_concept,
            bottlenecks,
            sess,
            args.source_dir,
            activations_dir,
            cavs_dir,
            num_random_exp=args.num_random_exp,
            channel_mean=True,
            max_imgs=args.max_imgs,
            min_imgs=args.min_imgs,
            num_discovery_imgs=args.max_imgs,
            num_workers=args.num_parallel_workers)
        # Creating the dataset of image patches

        print(datetime.datetime.now(), 'Creating the dataset of image patches of {}...'.format(target_class))
        cd.create_patches(param_dict={'n_segments': [15, 50, 80]}, gradcam=args.use_gradcam,
        #cd.create_patches(param_dict={'n_segments': [5, 15, 50]}, gradcam=args.use_gradcam,
                          keep_percent=args.keep_percent)
        '''for X-ray, we need larget patch'''

        # Saving the concept discovery target class images
        image_dir = os.path.join(discovered_concepts_dir, 'images')
        tf.io.gfile.makedirs(image_dir)
        ace_helpers.save_images(image_dir,
                                (cd.discovery_images * 256).astype(np.uint8))
        # Discovering Concepts
        print(datetime.datetime.now(), 'Discovering concepts of {}...'.format(target_class))
        cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})

        del cd.dataset  # Free memory
        del cd.image_numbers
        del cd.patches

        '''
        Save the concept information as dict
        '''
        # Save as txt
        dict_name = cd.dic
        dict_save_path = os.path.join(discovered_concepts_dir, 'all_concept_dict_X.txt')
        f = open(dict_save_path, 'wb')
        pickle.dump(dict_name, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        # Save discovered concept images (resized and original sized)
        ace_helpers.save_concepts(cd, discovered_concepts_dir)
        # Calculating CAVs and TCAV scores
        print(datetime.datetime.now(), 'Calculating CAVs and TCAV scores of {}...'.format(target_class))
        cav_accuraciess = cd.cavs(min_acc=0.0)
        scores = cd.tcavs(test=False)
        discover_concept_dic=[]

        result=ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
                                    results_summaries_dir + 'ace_results.txt')



        # Plot examples of discovered concepts
        print(datetime.datetime.now(), 'Plotting examples of discovered concepts of {}...'.format(target_class))
        for bn in cd.bottlenecks:
            print("bn is {}".format(bn))
            print("cd is {}".format(cd))
            ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)
            # Delete concepts that don't pass statistical testing
            cd.test_and_remove_concepts(scores)
        print('finish discovering concept of {}'.format(target_class))
        discover_concept_dic = discover_concept_dic.append(result)
    #return discover_concept_dic
    print(discover_concept_dic)


def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='source',
                        help='''Directory where the network's classes image folders and random concept folders are saved.''')
    parser.add_argument('--working_dir', type=str,
                        help='Directory to save the results.', default='result')

    parser.add_argument('--use_gradcam', type=bool,
                        help='''Whether use gradcam to filter the patches.''', default=True)
    parser.add_argument('--gradcam_layer', type=str, help='''which layer to use gradcam.''', default='')
    parser.add_argument('--keep_percent', type=int,
                        help='''the percentage of gradcam to filter the mask.''', default=50)

    parser.add_argument('--model_to_run', type=str,
                        help='The name of the model.', default='Resnet152')
    parser.add_argument('--model_path', type=str,
                        help='Path to model checkpoints.', default='')
    parser.add_argument('--labels_path', type=str,
                        help='Path to model checkpoints.', default='src/Xception_labels.json')

    '''Because ACE only have Imagenet classes, so you should copy and past Chexpert
  images into zebra'''
    parser.add_argument('--target_class', type=str,
                        help='The name of the target class to be interpreted,zebra, ambulance', default='all')
    parser.add_argument('--bottlenecks', type=str,
                        help='Names of the target layers of the network (comma separated)',
                        default='')
    parser.add_argument('--num_random_exp', type=int,
                        help="Number of random experiments used for statistical testing, etc",
                        default=20)
    parser.add_argument('--max_imgs', type=int,
                        help="Maximum number of images in a discovered concept",
                        default=50)
    parser.add_argument('--min_imgs', type=int,
                        help="Minimum number of images in a discovered concept",
                        default=40)
    parser.add_argument('--num_parallel_workers', type=int,
                        help="Number of parallel jobs.",
                        default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

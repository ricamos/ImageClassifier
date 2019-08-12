# Imports
import argparse
import util
import os, random
import json
import torch

#Configure log
logging = util.setup_logger(__name__, 'app.log') 

# Configure ArgumentParser 
parser = argparse.ArgumentParser(description = 'Predict the different species of flowers.')

parser.add_argument('img_path', action = 'store', help = 'Directory with images for predict.')
parser.add_argument('checkpoint_file', action = 'store', help = 'checkpoint file.')
parser.add_argument('--gpu', action='store_true', help='use gpu to infer classes')
parser.add_argument('--topk', action = 'store', dest = 'topk', type=int, default = 5, required = False, help = 'Return top K most likely classes')
parser.add_argument('--category_names', action='store', help='Label mapping file')

arguments = parser.parse_args()

try:
    # Use GPU if it's available
    #device = util.choose_device(arguments.gpu)
    
    #loads a checkpoint and rebuilds the model
    model = util.load_checkpoint(arguments.checkpoint_file)
    model.eval()
    
    #Image Preprocessing
    img_file = random.choice(os.listdir(arguments.img_path))
    image_path = arguments.img_path+img_file
    img = util.process_image(image_path)
    
    # Class Prediction
    probs, classes = util.predict(image_path, model, arguments.gpu, arguments.topk)
 
    # Sanity Checking
    cat_to_name = util.cat_to_name(classes, model, arguments.category_names)
    
    for i in range(len(cat_to_name)):
        print(f"class = {cat_to_name[i]} prob = {probs.data[0][i]:.3f}")
    #util.view_classify(image_path, probs, classes, cat_to_name)   
except Exception as e:
    logging.exception("Exception occurred")
    
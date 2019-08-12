# Imports
import argparse
import util

from workspace_utils import active_session

#Configure log
logging = util.setup_logger(__name__, 'app.log') 
logging.info('Inicio da execução')

# Configure ArgumentParser 
parser = argparse.ArgumentParser(description = 'Train an AI model.')

parser.add_argument('data_directory', action = 'store', help = 'Images directory.')
parser.add_argument('--batch_size', action = 'store', dest = 'batch_size', type=int, default = 32, required = False, help = 'batch size')
parser.add_argument('--arch', action = 'store', dest = 'arch', type=str, default = 'vgg19_bn', required = False,
                           help = "Choose architecture: The tourchvision model architectures for image classification. \
                           https://pytorch.org/docs/master/torchvision/models.html")
parser.add_argument('--output_size', action = 'store', dest = 'output_size', type=int, default = 102, required = False, help = 'Number of classes for output')
parser.add_argument('--hidden_units', action = 'store', dest = 'hidden_units', type=int, default = 1024, required = False, help = 'Set hyperparameters: Hidden layer')
parser.add_argument('--learning_rate', action = 'store', dest = 'learning_rate', type=float, default = 0.003, required = False, help = 'Set hyperparameters: The lerning rate')
parser.add_argument('--epochs', action = 'store', dest = 'epochs', type=int, default = 8, required = False, help = 'Set hyperparameters: Epochs number ')
parser.add_argument('--gpu', action = 'store_true', dest = 'gpu', help = 'Use GPU for training.')
parser.add_argument('--save_dir', action = 'store', dest = 'save_dir', type=str, default = None, required = False, help = 'Set directory to save checkpoints.')

arguments = parser.parse_args()

try:
    with active_session():
        # Load the data
        data_directory = arguments.data_directory
        train_data, test_data, valid_data, trainloader, testloader, validloader = util.load_data(data_directory, arguments.batch_size)
        logging.info('O diretorio foi configurado com sucesso!')

        # TODO: Load a pre-trained network and configure
        model, criterion, optimizer, classifier = util.load_model(arguments.arch, arguments.hidden_units, arguments.output_size, arguments.learning_rate)
        print(model, criterion, optimizer)
        logging.info('Modelo configurado com sucesso!')

        #Use GPU if it's available
        device = util.choose_device(arguments.gpu)
        print(device)
        logging.info("%s selecionado.",device )
        
        # TODO: Train de model
        logging.info('Inicio do treinamento!')
        util.train_model(model, trainloader, validloader, arguments.epochs, criterion, optimizer, device)
        logging.info('Modelo treinado com sucesso!')
        
        # TODO: Do validation on the test set
        accuracy = util.accuracy_network(model, testloader, criterion, device)
        logging.info(f'Acurracy foi medido em {accuracy} com sucesso!')
        
        # TODO: Save the checkpoint 
        file = util.save_checkpoint(model, train_data, optimizer, arguments.save_dir, arguments.arch, arguments.output_size, classifier, arguments.learning_rate, arguments.batch_size, arguments.epochs)
        logging.info("Arquivo checkpoint %s foi salvo.", file )
        logging.info('Fim da configuração do modelo. Parabéns!!!')
        
except Exception as e:
    logging.exception("Exception occurred")
 
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return input_image, keep, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    l3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', 
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), name = 'layer3conv1x1')
    l4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', 
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), name = 'layer4conv1x1')
    l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', 
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), name = 'layer7conv1x1')
    
    decoderlayer1 = tf.layers.conv2d_transpose(l7_conv_1x1, num_classes, 4, 2, name = 'decoderlayer1', 
                                                padding='same', 
                                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    decoderlayer2 = tf.add(decoderlayer1, l4_conv_1x1, name = 'decoderlayer2')
                            #kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    decoderlayer3 = tf.layers.conv2d_transpose(decoderlayer2, num_classes, 4, 2, name = 'decoderlayer3', 
                                                padding='same', 
                                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    decoderlayer4 = tf.add(decoderlayer3, l3_conv_1x1, name = 'decoderlayer4') 
                            #kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    output = tf.layers.conv2d_transpose(decoderlayer4, num_classes, 16, 8, name = 'output',
                                        padding='same', 
                                        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))


    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    class_labels = tf.reshape(correct_label, (-1, num_classes))

    # The cross_entropy_loss is the cost which we are trying to minimize to yield higher accuracy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = class_labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for epoch in epochs:

        losses, i = [], 0

        for images, labels in get_batches_fn(batch_size):

            i += 1

            feed = { input_image: images,
               correct_label: labels,
               keep_prob: keep_prob,
               learning_rate: learning_rate }
            _, partial_loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)

            print("---> iteration: ", i, " partial loss:", partial_loss)
            losses.append(partial_loss)
          
        training_loss = sum(losses) / len(losses)
        all_training_losses.append(training_loss)
    
        print("------------------")
        print("epoch: ", epoch + 1, " of ", epochs, "training loss: ", training_loss)
        print("------------------")

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    epochs = 20
    batch_size = 1
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        # Returns the three layers, keep probability and input layer from the vgg architecture
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)

        # The resulting network architecture from adding a decoder on top of the given vgg model
        model_output = layers(layer3, layer4, layer7, num_classes)

        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Train the neural network
        train_nn(session, epochs, batch_size, get_batches_fn, 
                train_op, cross_entropy_loss, image_input,
                correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        # Run the model with the test images and save each painted output image (roads painted green)
        helper.save_inference_samples(runs_dir, data_dir, session, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

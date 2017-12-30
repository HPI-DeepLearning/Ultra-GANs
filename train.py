from keras.optimizers import Adam

from model import *
from utils import *
from config import *

# Create optimizers
opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  
gen = Generator((sequence_length, size, size, input), output, kernel_depth)
gen.compile(loss='mae', optimizer=opt_discriminator)

disc = Discriminator((sequence_total, size, size, input), (sequence_total, size, size, output), kernel_depth)
disc.trainable = False

combined = Combine(gen, disc, (sequence_length, size, size, input))
loss = ['categorical_crossentropy', 'binary_crossentropy']
loss_weights = [5, 1]
combined.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

disc.trainable = True
disc.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

if os.path.isfile(checkpoint_gen_name):
    gen.load_weights(checkpoint_gen_name)
if os.path.isfile(checkpoint_disc_name):
    disc.load_weights(checkpoint_disc_name)

# List sequences  
sequences = prepare_data(data_dir)

real_y = np.reshape(np.array([0, 1]), (1, 2))
fake_y = np.reshape(np.array([1, 0]), (1, 2))

for e in range(epochs):
    print("Epoch {}".format(e))
    random.shuffle(sequences)
    
    progbar = keras.utils.Progbar(len(sequences))
    
    for s in range(len(sequences)):
        
        progbar.add(1)
        sequence = sequences[s]
        x, y = load(sequence, sequence_length)
        
        for i in range(len(x)):
        
            # train disc on real
            disc.train_on_batch([strip(x[i]), y[i]], real_y)
        
            # gen fake
            fake = gen.predict(x[i])
        
            # train disc on fake
            disc.train_on_batch([strip(x[i]), re_shape(fake)], fake_y)
        
            # train combined    
            disc.trainable = False
            combined.train_on_batch(x[i], [np.reshape(y[i], (1, 4*240*240, 4)), real_y])
            disc.trainable = True
            
        # output random result
        random_index = random.randrange(0,len(x))
        generated_y = gen.predict(x[random_index])
        save_image(strip(x[random_index]) / 2 + 0.5, y[random_index], re_shape(generated_y), "validation/e{}_{}.png".format(e, s))
        
    # save weights
    gen.save_weights(checkpoint_gen_name, overwrite=True)
    disc.save_weights(checkpoint_disc_name, overwrite=True)
        
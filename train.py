import tensorflow as tf
import os
from tensorflow.keras import optimizers, callbacks
from models.generator import build_unet_generator
from data.dataset import get_dataset
from utils.image_processing import denormalize_image, adaptive_histogram_equalization
from config import *
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def fft_loss(y_true, y_pred):
    # Compute FFT loss
    y_true = tf.cond(tf.equal(tf.rank(y_true), 5),
                     lambda: tf.reduce_mean(y_true, axis=-1),
                     lambda: y_true)
    y_pred = tf.cond(tf.equal(tf.rank(y_pred), 5),
                     lambda: tf.reduce_mean(y_pred, axis=-1),
                     lambda: y_pred)

    fft_true = tf.signal.fft2d(tf.cast(y_true, tf.complex64))
    fft_pred = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))
    
    magnitude_true = tf.abs(fft_true)
    magnitude_pred = tf.abs(fft_pred)
    
    log_true = tf.math.log(magnitude_true + 1e-8)
    log_pred = tf.math.log(magnitude_pred + 1e-8)
    
    mse = tf.reduce_mean(tf.square(log_true - log_pred))
    
    return mse

def create_cross_mask(shape):
    # Create a cross-shaped mask
    batch = shape[0]
    depth = shape[1]
    height = shape[2]
    width = shape[3]

    mask = tf.zeros(shape)
    center = tf.cast(tf.stack([height // 2, width // 2]), tf.int32)
    mask_width = tf.cast(tf.stack([height // 50, width // 50]), tf.int32)
    
    vertical_mask = tf.pad(
        tf.ones([depth, height, mask_width[1]]),
        [[0, 0], [0, 0], 
         [center[1] - mask_width[1] // 2, width - center[1] - mask_width[1] // 2]]
    )
    vertical_mask = tf.expand_dims(vertical_mask, 0)
    vertical_mask = tf.tile(vertical_mask, [batch, 1, 1, 1])
    
    horizontal_mask = tf.pad(
        tf.ones([depth, mask_width[0], width]),
        [[0, 0],
         [center[0] - mask_width[0] // 2, height - center[0] - mask_width[0] // 2],
         [0, 0]]
    )
    horizontal_mask = tf.expand_dims(horizontal_mask, 0)
    horizontal_mask = tf.tile(horizontal_mask, [batch, 1, 1, 1])
    
    mask = mask + vertical_mask + horizontal_mask
    return mask

class Noise2Noise(tf.keras.Model):
    def __init__(self, input_shape):
        # Initialize Noise2Noise model
        super(Noise2Noise, self).__init__()
        self.generator = build_unet_generator(input_shape, GENERATOR_FILTERS)
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.ssim_loss = tf.image.ssim
        self.l1_loss = tf.keras.losses.MeanAbsoluteError()

    def compile(self, optimizer='adam', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        super(Noise2Noise, self).compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, jit_compile=jit_compile, **kwargs)

    @tf.function
    def step(self, data, training=True):
        # Perform a single training or validation step
        noisy_images1, noisy_images2 = data
        
        noisy_images1, noisy_images2 = self._adjust_input_shape(noisy_images1, noisy_images2)
        
        if training:
            with tf.GradientTape() as tape:
                denoised_images = self.generator(noisy_images1, training=True)
                losses = self._compute_losses(noisy_images2, denoised_images)
                total_loss = (SSIM_WEIGHT * losses['ssim_loss'] + 
                              L1_WEIGHT * losses['l1_loss'] + 
                              GRAD_WEIGHT * losses['grad_loss'] + 
                              NUCLEUS_WEIGHT * losses['nucleus_loss'] +
                              FFT_WEIGHT * losses['fft_loss'])
            
            gradients = tape.gradient(total_loss, self.generator.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        else:
            denoised_images = self.generator(noisy_images1, training=False)
            losses = self._compute_losses(noisy_images2, denoised_images)
            total_loss = (SSIM_WEIGHT * losses['ssim_loss'] + 
                          L1_WEIGHT * losses['l1_loss'] + 
                          GRAD_WEIGHT * losses['grad_loss'] + 
                          NUCLEUS_WEIGHT * losses['nucleus_loss'] +
                          FFT_WEIGHT * losses['fft_loss'])
        
        
        psnr = tf.reduce_mean(tf.image.psnr(noisy_images2, denoised_images, max_val=1.0))
        losses.update({"total_loss": total_loss, "psnr": psnr})
        return losses

    def _adjust_input_shape(self, noisy_images1, noisy_images2):
        # Adjust input shape if necessary
        if len(noisy_images1.shape) == 6:
            noisy_images1 = tf.reshape(noisy_images1, (-1,) + PATCH_SIZE + (1,))
            noisy_images2 = tf.reshape(noisy_images2, (-1,) + PATCH_SIZE + (1,))
        elif len(noisy_images1.shape) == 5 and noisy_images1.shape[-1] != 1:
            noisy_images1 = tf.expand_dims(noisy_images1, axis=-1)
            noisy_images2 = tf.expand_dims(noisy_images2, axis=-1)
        return noisy_images1, noisy_images2

    def _compute_losses(self, y_true, y_pred):
        # Compute various loss components
        ssim_loss = 1 - tf.reduce_mean(self.ssim_loss(y_true, y_pred, max_val=1.0))
        l1_loss = self.l1_loss(y_true, y_pred)
        grad_loss = self.gradient_loss(y_true, y_pred)
        nucleus_loss = self.nucleus_feature_loss(y_true, y_pred)
        fft_loss_value = fft_loss(y_true, y_pred)
        
        return {
            "ssim_loss": ssim_loss,
            "l1_loss": l1_loss,
            "grad_loss": grad_loss,
            "nucleus_loss": nucleus_loss,
            "fft_loss": fft_loss_value
        }

    def gradient_loss(self, y_true, y_pred):
        # Compute gradient loss
        def compute_gradient(x):
            dy = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
            dy = tf.pad(dy, [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]])
            
            dx = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
            dx = tf.pad(dx, [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0]])
            
            dz = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]
            dz = tf.pad(dz, [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]])
            
            return dy, dx, dz

        dy_true, dx_true, dz_true = compute_gradient(y_true)
        dy_pred, dx_pred, dz_pred = compute_gradient(y_pred)
        
        return tf.reduce_mean(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred) + tf.abs(dz_true - dz_pred))

    def nucleus_feature_loss(self, y_true, y_pred):
        # Compute nucleus feature loss
        pool_size = (1, 2, 2, 2, 1)
        strides = (1, 1, 1, 1, 1)
        y_true_max = tf.nn.max_pool3d(y_true, pool_size, strides=strides, padding='SAME')
        y_pred_max = tf.nn.max_pool3d(y_pred, pool_size, strides=strides, padding='SAME')
        
        y_true_edges = self.edge_detection(y_true)
        y_pred_edges = self.edge_detection(y_pred)
        
        return tf.reduce_mean(tf.abs(y_true_max - y_pred_max)) + 0.5 * tf.reduce_mean(tf.abs(y_true_edges - y_pred_edges))

    def edge_detection(self, x):
        # Perform simple 3D edge detection
        kernel = tf.constant([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                             [[2, 4, 2], [4, -84, 4], [2, 4, 2]],
                             [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=tf.float32)
        kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)
        return tf.nn.conv3d(x, kernel, strides=[1,1,1,1,1], padding='SAME')

class VisualizeDenoising(callbacks.Callback):
    def __init__(self, val_dataset, log_dir, visualization_interval):
        # Initialize visualization callback
        super(VisualizeDenoising, self).__init__()
        self.val_dataset = val_dataset
        self.log_dir = log_dir
        self.visualization_interval = visualization_interval
        self._model = None

    def set_model(self, model):
        self._model = model

    def on_epoch_end(self, epoch, logs=None):
        # Visualize denoising results at the end of each epoch
        if (epoch + 1) % self.visualization_interval == 0:
            for noisy_images, clean_images in self.val_dataset.take(1):
                if len(noisy_images.shape) == 5:
                    noisy_images = tf.reshape(noisy_images, (-1,) + PATCH_SIZE + (1,))
                    clean_images = tf.reshape(clean_images, (-1,) + PATCH_SIZE + (1,))
                elif len(noisy_images.shape) == 6:
                    noisy_images = tf.reshape(noisy_images, (-1,) + PATCH_SIZE + (1,))
                    clean_images = tf.reshape(clean_images, (-1,) + PATCH_SIZE + (1,))
                
                noisy_images = noisy_images[:1]
                clean_images = clean_images[:1]
                
                denoised_images = self._model.generator(noisy_images, training=False)
                
                z_middle = PATCH_SIZE[0] // 2
                noisy_slice = noisy_images[0, z_middle, :, :, 0].numpy()
                clean_slice = clean_images[0, z_middle, :, :, 0].numpy()
                denoised_slice = denoised_images[0, z_middle, :, :, 0].numpy()

            

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(denormalize_image(noisy_slice), cmap='gray', vmin=0, vmax=255)
                axes[0].set_title('Noisy (Middle XY Plane)')
                axes[0].axis('off')
                
                axes[1].imshow(denormalize_image(clean_slice), cmap='gray', vmin=0, vmax=255)
                axes[1].set_title('Clean (Middle XY Plane)')
                axes[1].axis('off')
                
                axes[2].imshow(denormalize_image(denoised_slice), cmap='gray', vmin=0, vmax=255)
                axes[2].set_title('Denoised (Middle XY Plane)')
                axes[2].axis('off')

                plt.suptitle(f'Epoch {epoch + 1}')
                plt.tight_layout()
                
                save_path = os.path.join(self.log_dir, f'denoising_result_epoch_{epoch + 1}.png')
                plt.savefig(save_path)
                plt.close()

                np.save(os.path.join(self.log_dir, f'noisy_slice_epoch_{epoch + 1}.npy'), noisy_slice)
                np.save(os.path.join(self.log_dir, f'clean_slice_epoch_{epoch + 1}.npy'), clean_slice)
                np.save(os.path.join(self.log_dir, f'denoised_slice_epoch_{epoch + 1}.npy'), denoised_slice)

class CustomTensorBoard(callbacks.TensorBoard):
    def __init__(self, **kwargs):
        # Initialize custom TensorBoard callback
        super().__init__(**kwargs)
        self._custom_model = None
        self._custom_optimizer = None

    def set_model(self, model):
        super().set_model(model)
        self._custom_model = model
        self._custom_optimizer = model.optimizer

    def _collect_learning_rate(self, logs):
        if self._custom_optimizer:
            logs = logs or {}
            logs['learning_rate'] = self._custom_optimizer.learning_rate
        return logs

def get_callbacks(val_dataset, model, log_dir):
    # Get list of callbacks for training
    callbacks_list = [
        CustomTensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch',
            write_graph=True,
            write_images=True
        ),
        callbacks.EarlyStopping(
            monitor='val_total_loss', 
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='min'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_total_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='min'
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "noise2noise_best_{epoch:02d}.weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_total_loss',
            mode='min'
        ),
        VisualizeDenoising(val_dataset, log_dir, visualization_interval=VISUALIZATION_INTERVAL)
    ]
    
    for callback in callbacks_list:
        if hasattr(callback, 'set_model'):
            callback.set_model(model)
    
    return callbacks_list

def train():
    # Main training function
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)

    log_dir = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    train_dataset = get_dataset(is_training=True)
    val_dataset = get_dataset(is_training=False)
    
    model = Noise2Noise(PATCH_SIZE + (1,))
    model.compile()
    
    callbacks = get_callbacks(val_dataset, model, log_dir)
    
    total_steps = sum(1 for _ in train_dataset)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        progress_bar = tqdm(total=total_steps, unit='batch')
        epoch_losses = []
        for batch in train_dataset:
            logs = model.step(batch, training=True)
            epoch_losses.append(logs['total_loss'])
            
            progress_bar.update(1)
            progress_bar.set_postfix({k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in logs.items()})
        progress_bar.close()
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        val_logs = {}
        for batch in val_dataset:
            if len(batch[0].shape) == 5:
                batch = (tf.reshape(batch[0], (-1,) + PATCH_SIZE + (1,)),
                         tf.reshape(batch[1], (-1,) + PATCH_SIZE + (1,)))
            
            batch_val_logs = model.step(batch, training=False)
            for k, v in batch_val_logs.items():
                val_logs[k] = val_logs.get(k, 0) + v.numpy()
        
        val_logs = {k: v / len(val_dataset) for k, v in val_logs.items()}
        val_logs = {'val_' + k: v for k, v in val_logs.items()}
        
        print(f"Validation: {val_logs}")
        
        epoch_logs = {**logs, **val_logs}
        
        for callback in callbacks:
            callback.on_epoch_end(epoch, epoch_logs)
        
        if (epoch + 1) % VISUALIZATION_INTERVAL == 0:
            visualize_fft(model, val_dataset, epoch, log_dir)
    
    if MODEL_SAVE_FORMAT == "keras":
        model.generator.save(os.path.join(CHECKPOINT_DIR, "noise2noise_final_model.keras"))
    elif MODEL_SAVE_FORMAT == "h5":
        model.generator.save(os.path.join(CHECKPOINT_DIR, "noise2noise_final_model.h5"))
    else:
        raise ValueError(f"Unsupported model save format: {MODEL_SAVE_FORMAT}")
    
    model.generator.save_weights(os.path.join(CHECKPOINT_DIR, "noise2noise_final.weights.h5"))
    
    return None

def visualize_fft(model, dataset, epoch, log_dir):
    # Visualize FFT results
    for noisy_images, clean_images in dataset.take(1):
        if len(noisy_images.shape) == 6:
            noisy_images = tf.reshape(noisy_images, (-1, 64, 128, 128, 1))
            clean_images = tf.reshape(clean_images, (-1, 64, 128, 128, 1))
        elif len(noisy_images.shape) == 5 and noisy_images.shape[1] == 2:
            noisy_images = tf.reshape(noisy_images, (-1, 64, 128, 128, 1))
            clean_images = tf.reshape(clean_images, (-1, 64, 128, 128, 1))
        elif len(noisy_images.shape) == 4:
            noisy_images = tf.expand_dims(noisy_images, axis=-1)
            clean_images = tf.expand_dims(clean_images, axis=-1)
        elif len(noisy_images.shape) == 3:
            noisy_images = tf.expand_dims(tf.expand_dims(noisy_images, axis=0), axis=-1)
            clean_images = tf.expand_dims(tf.expand_dims(clean_images, axis=0), axis=-1)
        
        denoised_images = model.generator(noisy_images, training=False)

        noisy_slice = noisy_images[0, VISUALIZATION_SLICE, :, :, 0].numpy()
        clean_slice = clean_images[0, VISUALIZATION_SLICE, :, :, 0].numpy()
        denoised_slice = denoised_images[0, VISUALIZATION_SLICE, :, :, 0].numpy()

        fft_noisy = np.fft.fftshift(np.fft.fft2(noisy_slice))
        fft_clean = np.fft.fftshift(np.fft.fft2(clean_slice))
        fft_denoised = np.fft.fftshift(np.fft.fft2(denoised_slice))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.log(np.abs(fft_noisy) + 1), cmap='gray')
        axes[0].set_title('Noisy FFT (Slice)')
        axes[1].imshow(np.log(np.abs(fft_clean) + 1), cmap='gray')
        axes[1].set_title('Clean FFT (Slice)')
        axes[2].imshow(np.log(np.abs(fft_denoised) + 1), cmap='gray')
        axes[2].set_title('Denoised FFT (Slice)')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'fft_visualization_epoch_{epoch+1}.png'))
        plt.close()

if __name__ == "__main__":
    train()
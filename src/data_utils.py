import tensorflow as tf
import numpy as np

class TF_Macenko_Normalizer:
    """
    GPU-accelerated Macenko stain normalization using pure TensorFlow operations.

    This implementation converts all NumPy operations to TensorFlow equivalents,
    allowing the normalization to execute on GPU and be compiled with @tf.function.

    Key improvements over HED_Macenko_Normalizer:
    - Pure TensorFlow operations (no NumPy)
    - GPU-compatible (can run on CUDA devices)
    - Compatible with tf.data pipeline without py_function wrapper
    - Supports batch processing

    References:
    - Macenko, M., et al. (2009). "A method for normalizing histology slides for
      quantitative analysis." IEEE International Symposium on Biomedical Imaging.
    """

    def __init__(self):
        """Initialize with reference stain vectors for H&E."""
        # Standard reference values (Hematoxylin and Eosin stain vectors)
        self.HERef = tf.constant([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ], dtype=tf.float32)
        self.maxCRef = tf.constant([1.9705, 1.0308], dtype=tf.float32)
    
    def fit(self, I, beta=0.15, alpha=1.0):
        """
        Fit the normalizer to a reference image by extracting its stain matrix.
        
        This method extracts the stain vectors and maximum concentrations from
        a reference image and stores them for use in normalization.
        
        Args:
            I: Reference image as NumPy array [H, W, 3] in range [0, 255]
            beta: OD threshold for filtering (default 0.15)
            alpha: Percentile threshold (default 1.0)
        """
        # Convert NumPy array to TensorFlow tensor if needed
        if not isinstance(I, tf.Tensor):
            I = tf.constant(I, dtype=tf.float32)
        
        # Extract stain matrix from reference image
        stain_matrix = self._get_stain_matrix(I, beta, alpha)
        
        if stain_matrix is not None:
            # Update reference stain matrix
            self.HERef = stain_matrix
            
            # Get concentrations for reference image
            C = self._get_concentrations(I, stain_matrix)
            
            # Calculate max concentrations
            maxC_0 = self._tf_percentile(C[:, 0], 99.0)
            maxC_1 = self._tf_percentile(C[:, 1], 99.0)
            self.maxCRef = tf.stack([maxC_0, maxC_1])
            
            print(f"Macenko normalizer fitted successfully")
        else:
            print("Warning: Could not extract stain matrix from reference image. Using default values.")
            # Keep default values

    def _tf_percentile(self, data, q):
        """
        TensorFlow implementation of percentile calculation.

        Args:
            data: 1D tensor
            q: Percentile value (0-100)

        Returns:
            Percentile value as scalar tensor
        """
        k = tf.cast(tf.cast(tf.size(data), tf.float32) * (q / 100.0), tf.int32)
        k = tf.maximum(k, 1)
        k = tf.minimum(k, tf.size(data))
        sorted_data = tf.sort(data)
        return sorted_data[k - 1]

    def _get_stain_matrix(self, I, beta=0.15, alpha=1.0):
        """
        Compute the stain matrix using TensorFlow operations.

        Args:
            I: Input image tensor [H, W, 3] in range [0, 255]
            beta: OD threshold for filtering
            alpha: Percentile threshold

        Returns:
            Stain matrix [3, 2] or None if insufficient valid pixels
        """
        # Convert to Optical Density (OD)
        I_float = tf.cast(I, tf.float32)
        OD = -tf.math.log((I_float + 1.0) / 255.0)

        # Reshape and filter low OD values
        h, w, c = tf.unstack(tf.shape(OD))
        ODhat = tf.reshape(OD, [-1, 3])

        # Remove pixels with any channel below beta threshold
        mask = tf.reduce_all(ODhat >= beta, axis=1)
        ODhat_filtered = tf.boolean_mask(ODhat, mask)

        # Check if we have enough valid pixels
        n_pixels = tf.shape(ODhat_filtered)[0]
        if n_pixels < 10:
            return None

        # Calculate covariance matrix: Cov = (X^T @ X) / (n - 1)
        mean = tf.reduce_mean(ODhat_filtered, axis=0, keepdims=True)
        centered = ODhat_filtered - mean
        n = tf.cast(tf.shape(centered)[0], tf.float32)
        cov_matrix = tf.matmul(centered, centered, transpose_a=True) / (n - 1.0)

        # Calculate eigenvectors (eigenvalues, eigenvectors)
        eigenvalues, eigenvectors = tf.linalg.eigh(cov_matrix)

        # Take eigenvectors corresponding to two largest eigenvalues
        # tf.linalg.eigh returns eigenvalues in ascending order
        Vec = eigenvectors[:, 1:3]  # Last 2 eigenvectors

        # Project data onto 2D plane
        That = tf.matmul(ODhat_filtered, Vec)

        # Calculate angles
        phi = tf.math.atan2(That[:, 1], That[:, 0])

        # Find min and max angles using percentiles
        minPhi = self._tf_percentile(phi, alpha)
        maxPhi = self._tf_percentile(phi, 100.0 - alpha)

        # Calculate stain vectors
        vMin = tf.matmul(Vec, tf.stack([tf.cos(minPhi), tf.sin(minPhi)], axis=0)[:, None])
        vMax = tf.matmul(Vec, tf.stack([tf.cos(maxPhi), tf.sin(maxPhi)], axis=0)[:, None])

        vMin = tf.squeeze(vMin)
        vMax = tf.squeeze(vMax)

        # Order vectors (Hematoxylin first, Eosin second)
        if vMin[0] > vMax[0]:
            HE = tf.stack([vMin, vMax], axis=0)
        else:
            HE = tf.stack([vMax, vMin], axis=0)

        return tf.transpose(HE)  # Return [3, 2]

    def _get_concentrations(self, I, stain_matrix):
        """
        Get stain concentrations using TensorFlow least squares.

        Args:
            I: Input image [H, W, 3]
            stain_matrix: Stain matrix [3, 2]

        Returns:
            Concentration matrix [N, 2] where N = H * W
        """
        # Convert to OD
        I_float = tf.cast(I, tf.float32)
        OD = -tf.math.log((I_float + 1.0) / 255.0)

        h, w, c = tf.unstack(tf.shape(OD))
        OD_flat = tf.reshape(OD, [-1, 3])

        # Solve: stain_matrix @ C^T = OD^T
        # => C^T = lstsq(stain_matrix, OD^T)
        # => C = lstsq(stain_matrix, OD^T)^T
        C_T = tf.linalg.lstsq(stain_matrix, tf.transpose(OD_flat))
        C = tf.transpose(C_T)

        return C

    def normalize(self, I, Io=240.0, beta=0.15, alpha=1.0):
        """
        Normalize image using Macenko method (GPU-accelerated).

        Note: Not decorated with @tf.function at method level to allow conditional
        returns. Will be wrapped in tf.function when used in data pipeline.

        Args:
            I: Input image tensor [H, W, 3] in range [0, 255]
            Io: Intensity of light (default 240)
            beta: OD threshold
            alpha: Percentile threshold

        Returns:
            Normalized image [H, W, 3] in range [0, 255]
        """
        try:
            # Store original shape and dtype
            original_shape = tf.shape(I)
            original_dtype = I.dtype

            # Get stain matrix
            stain_matrix = self._get_stain_matrix(I, beta, alpha)

            # If stain extraction fails, return original image
            if stain_matrix is None:
                return I

            # Get concentrations
            C = self._get_concentrations(I, stain_matrix)

            # Normalize concentrations
            maxC_0 = self._tf_percentile(C[:, 0], 99.0)
            maxC_1 = self._tf_percentile(C[:, 1], 99.0)
            maxC = tf.stack([maxC_0, maxC_1])

            # Avoid division by zero
            tmp = maxC / (self.maxCRef + 1e-6)
            tmp = tf.maximum(tmp, 1e-6)
            C2 = C / tmp

            # Reconstruct image
            # Inorm = Io * exp(-HERef @ C2^T)
            exponent = -tf.matmul(self.HERef, tf.transpose(C2))
            Inorm_flat = Io * tf.exp(exponent)
            Inorm_flat = tf.transpose(Inorm_flat)

            # Reshape to original image dimensions
            h, w, c = original_shape[0], original_shape[1], original_shape[2]
            Inorm = tf.reshape(Inorm_flat, [h, w, 3])

            # Clip and convert back to uint8
            Inorm = tf.clip_by_value(Inorm, 0.0, 255.0)
            Inorm = tf.cast(Inorm, original_dtype)

            return Inorm
        except Exception as e:
            # Fallback to original image if normalization fails
            tf.print("GPU Macenko normalization failed, returning original image:", e)
            return I


class HED_Macenko_Normalizer:
    """
    Implements stain normalization using the Macenko method.
    
    This class provides functionality to normalize the color distribution of histology images
    to a reference standard, reducing batch effects and staining variations.
    
    References:
    - Macenko, M., et al. (2009). "A method for normalizing histology slides for quantitative analysis."
      IEEE International Symposium on Biomedical Imaging: From Nano to Macro.
    """
    
    def __init__(self):
        # Standard reference values (from the paper or typical H&E standards)
        # These represent the stain vectors for Hematoxylin and Eosin
        self.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        self.maxCRef = np.array([1.9705, 1.0308])
        
    def _get_stain_matrix(self, I, beta=0.15, alpha=1):
        """
        Compute the stain matrix for an image I.
        """
        # Convert to OD
        OD = -np.log((I.astype(np.float64) + 1) / 255)
        
        # Remove data with too low OD
        ODhat = OD.reshape((-1, 3))
        mask = np.any(ODhat < beta, axis=1)
        ODhat = ODhat[~mask]
        
        if ODhat.shape[0] < 10:
            return None
            
        # Calculate eigenvectors
        _, V = np.linalg.eigh(np.cov(ODhat, rowvar=False))
        
        # Project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        Vec = V[:, 1:3]
        
        # Project data
        That = np.dot(ODhat, Vec)
        
        # Calculate angle of each point
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        # Find min and max vectors
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        
        vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        # Heuristic to order the vectors (Hematoxylin first, Eosin second)
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax])
        else:
            HE = np.array([vMax, vMin])
            
        return HE.T

    def _get_concentrations(self, I, stain_matrix):
        """
        Get stain concentrations.
        """
        OD = -np.log((I.astype(np.float64) + 1) / 255)
        OD = OD.reshape((-1, 3))
        
        # C = V^-1 * OD^T
        # We use least squares
        C = np.linalg.lstsq(stain_matrix, OD.T, rcond=None)[0]
        return C.T

    def normalize(self, I, Io=240, beta=0.15, alpha=1):
        """
        Normalize the image I.
        """
        try:
            h, w, c = I.shape
            stain_matrix = self._get_stain_matrix(I, beta, alpha)
            
            if stain_matrix is None:
                return I
                
            # Get concentrations
            C = self._get_concentrations(I, stain_matrix)
            
            # Normalize concentrations
            maxC = np.percentile(C, 99, axis=0)
            tmp = np.divide(maxC, self.maxCRef)
            C2 = np.divide(C, tmp)
            
            # Reconstruct
            Inorm = np.multiply(Io, np.exp(-np.dot(self.HERef, C2.T)))
            Inorm = Inorm.T.reshape((h, w, c))
            Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)
            
            return Inorm
        except Exception as e:
            # Fallback if normalization fails
            print(f"Normalization failed: {e}")
            return I

    def tf_normalize(self, image):
        """
        TensorFlow-compatible wrapper for normalization.
        """
        def _norm_func(img):
            return self.normalize(img)
            
        return tf.numpy_function(_norm_func, [image], tf.uint8)

def get_data_augmentation_pipeline(height=224, width=224):
    """
    Returns a Sequential model for data augmentation suitable for histopathology.
    Includes geometric transformations and color adjustments.
    """
    data_augmentation = tf.keras.Sequential([
        # Geometric augmentations
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        
        # Color augmentations (Subtle for histopathology to avoid destroying stain info)
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ], name="data_augmentation")
    
    return data_augmentation

def load_and_preprocess_image(path, label, img_height=224, img_width=224, normalizer=None):
    """
    Loads and preprocesses a single image.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    
    # Apply normalization if provided
    if normalizer:
        # Cast to uint8 for normalization
        image = tf.cast(image, tf.uint8)
        image = normalizer.tf_normalize(image)
        image = tf.cast(image, tf.float32)
    
    # Rescale to [0, 1] or [-1, 1] depending on model requirements
    # Here we stick to [0, 255] -> [0, 1] generic, but specific models might need preprocess_input
    # We will handle model-specific preprocessing in the notebook pipeline
    
    return image, label

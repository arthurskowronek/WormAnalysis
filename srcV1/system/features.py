"""
Feature extraction from images.
"""
import pyfeats
import warnings
import numpy as np
import pandas as pd
import scipy.ndimage
import networkx as nx
import skimage as ski
import matplotlib.pyplot as plt
from typing import List
from skimage import draw
from boruta import BorutaPy
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class FeatureExtractor:
    """Class for extracting features from images."""
        
    def __init__(self, dataset=None):
        """
        Initialize the FeatureExtractor with an optional dataset.
        
        Args:
            dataset: numpy Dataset to extract features from
        """
        self.feature_names = []
        self.set_feature_names()
        
        self.features = np.zeros((len(dataset), len(self.feature_names))) if dataset is not None else np.zeros((0, len(self.feature_names)))
        if dataset is not None:
            for i, item in enumerate(dataset):
                if item.coiled:
                    item.features = np.zeros(len(self.feature_names))
                else:
                    self.features[i] = item.features
      
    def get_features(self):
        """
        Get the features.
        """
        return self.features, self.feature_names

    def set_features(self, dataset):
        """
        Set the features.

        Args:
            dataset: Dataset object
        """      
        self.set_feature_names()
        
        images, _ = dataset.get_data()
        median_width = dataset.get_median_width()
        
        # Extract basic features
        basic_features = []
        for i, original_image in enumerate(images):
            print(f"Extracting features from image {i+1} of {len(images)}")
            # Get image features
            image_features = []
            region_features = []

            # create a mask from the image of the synapses
            coords_synapses = dataset.data[i].maxima
            mask_synapses = self._create_synapse_mask(original_image, coords_synapses)
            image = original_image * mask_synapses
            
            """# plot figure with original image and coordinates of synapses on image 1, mask of synapses on image 2 
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(original_image, cmap='gray')
            ax[0].set_title('Original Image with Synapses')
            # inverse x, y of coords_synapses to match the image coordinates
            coords_synapses2 = [(y, x) for x, y in coords_synapses]
            ax[0].scatter(*zip(*coords_synapses2), color='red', s=3, label='Synapses')
            ax[0].legend()
            ax[1].imshow(image, cmap='gray')
            ax[1].set_title('Mask of Synapses')
            plt.show()"""

            # get the mean intensity of the worm
            mask_worm = dataset.data[i].worm_mask
            mean_intensity = np.mean(image[mask_worm == 1])

            # get the regions of interest
            component, label_seg = self._get_regions_of_interest(coords_synapses, original_image, mask_synapses)
            
            #IMAGE_DIAPO
            """labeled_image = ski.color.label2rgb(label_seg, image=original_image, bg_label=0, alpha=0.5)
            
            # colorize the original_image with above regions of interest
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(original_image, cmap='gray')  
            ax.imshow(labeled_image, alpha=0.5)
            ax.set_title('Regions of Interest')
            plt.show()"""
            
            for prop in component:
                minr, minc, maxr, maxc = prop.bbox
                roi_mask = prop.image  # binary mask of the region
                roi_intensity = image[minr:maxr, minc:maxc]  # intensity values
                roi_intensity = roi_intensity - mean_intensity
                roi_intensity = np.clip(roi_intensity, 0, None)  # Ensure non-negative values   

                # Texture features
                texture_feat = self._extract_texture_features(roi_intensity, roi_mask, prop, median_width[i])
                region_features.append(texture_feat)
            
            # Agréger les caractéristiques de toutes les régions
            image_features.extend(self._aggregate_region_features(region_features))
            
            # Spatial features
            graph = dataset.data[i].graph
            spatial_feat = self._extract_spatial_features(graph, component)
            image_features.extend(spatial_feat) 

            # Add features
            measure_diff_slice = dataset.data[i].diff_slice
            measure_diff_segment = dataset.data[i].diff_segment
            image_features.extend([measure_diff_slice, measure_diff_segment, len(component)])

            # Add features to basic_features    
            basic_features.append(image_features)
           
        # Convert to numpy array
        basic_features = np.array(basic_features, dtype=np.float64)
        basic_features = np.nan_to_num(basic_features, nan=0.0, posinf=0.0, neginf=0.0)
            
        self.features = basic_features
    
    def set_feature_names(self):
        """
        Initialize feature names.
        """
        self.feature_names = []
        # Initialize feature names once
        # Texture feature names
        texture_base_names = []
        # FOS names
        fos_names = ['fos_' + str(i) for i in range(16)]
        texture_base_names.extend(fos_names)
        # NGTDM names
        ngtdm_names = ['ngtdm_' + str(i) for i in range(5)]
        texture_base_names.extend(ngtdm_names)
        # GLDS names
        glds_names = ['glds_' + str(i) for i in range(5)]
        texture_base_names.extend(glds_names)
        # SFM names
        sfm_names = ['sfm_' + str(i) for i in range(4)]
        texture_base_names.extend(sfm_names)
        # GLRLM names
        glrlm_names = ['glrlm_' + str(i) for i in range(11)]
        texture_base_names.extend(glrlm_names)
        # FPS names
        fps_names = ['fps_' + str(i) for i in range(2)]
        texture_base_names.extend(fps_names)
        # GLSZM names
        glszm_names = ['glszm_' + str(i) for i in range(14)]
        texture_base_names.extend(glszm_names)
        # Shape names
        shape_names = ['shape_' + str(i) for i in range(5)]
        texture_base_names.extend(shape_names)
        
        # Add statistics to feature names
        for name in texture_base_names:
            self.feature_names.append(f'{name}_mean')
            self.feature_names.append(f'{name}_std')
            if name.startswith('fos_') or name.startswith('ngtdm_'): 
                self.feature_names.append(f'{name}_min')
                self.feature_names.append(f'{name}_max')
                
        # Add spatial feature names
        spatial_feature_names = ['mean_nearest_neighbor_dist','std_nearest_neighbor_dist',
                                'min_nearest_neighbor_dist','max_nearest_neighbor_dist']
        spatial_feature_names.extend([f'ripley_k_{i}' for i in range(5)])
        
        # Add edge feature names
        edge_feature_names = ['mean_edge_length','max_edge_length','median_edge_length','min_edge_length','number_edges_sup_mean']
        spatial_feature_names.extend(edge_feature_names)
        self.feature_names.extend(spatial_feature_names)
        
        # Add measurement features
        measurement_feature_names = [
            'measure_diff_slice',
            'measure_diff_segment',
            'num_synapses'
        ]
        self.feature_names.extend(measurement_feature_names)

    def feature_reduction(self, y, indices_coiled):
        features, _, y = self._process_features(y, indices_coiled)         
        
        scaler = StandardScaler() 
        features = scaler.fit_transform(features)
        features = np.nan_to_num(features)
        
        pca = PCA(n_components=3)
        features = pca.fit_transform(features)
        
        
        # -------------------- plot PCA --------------------
        # 2D Plot
        plt.figure(figsize=(10, 10))
        color_map = {0: 'red', 1: 'blue'}
        colors = [color_map[label] for label in y]
        plt.scatter(
            features[:, 0],
            features[:, 1],
            c=colors,
            cmap='tab10',  # Better for categorical data (supports up to 10 classes)
            alpha=0.7
        )
        plt.title('Feature space with feature reduction')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()
        
        
        # 3D Plot
        fig = plt.figure(figsize=(12, 10))
        color_map = {0: 'red', 1: 'blue'}
        colors = [color_map[label] for label in y]
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            features[:, 0],
            features[:, 1],
            features[:, 2],
            c=colors,
            cmap='tab10',  # qualitative colormap for discrete labels
            alpha=0.6
        )
        # Axis labels
        ax.set_title('3D Feature Space with Feature Reduction')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.tight_layout()
        plt.show()
    
    def feature_selection(self, y, method='lasso', k=10, coiled_worms=None, verbose_features_selected=True):
        features, feature_names, y = self._process_features(y, coiled_worms) 
        # Remove features with all 0s
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        if method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=k)
            features = selector.fit_transform(features, y)
            selected_indices = selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
            # save indices of selected features in a file "selected_features.txt" in 'models' folder
            with open('models/selected_features.txt', 'w') as f:
                counter = 0
                for item in selected_indices:
                    if item == True:
                        # write the indices of the selected features
                        f.write("%s\n" % counter)
                    counter += 1
            if verbose_features_selected : print(f"Selected features (kbest): {selected_feature_names}")
            return features, selected_feature_names
        elif method == 'boruta':
            rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
            boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=42) 
            boruta_selector.fit(features, y)
            selected_indices = boruta_selector.support_
            selected_feature_names = [feature_names[i] for i, selected in enumerate(selected_indices) if selected]
            if verbose_features_selected : print(f"Selected features (boruta): {selected_feature_names}")
            # save indices of selected features in a file "selected_features.txt" in 'models' folder
            with open('models/selected_features.txt', 'w') as f:
                counter = 0
                for item in selected_indices:
                    if item == True:
                        # write the indices of the selected features
                        f.write("%s\n" % counter)
                    counter += 1
            return features[:, selected_indices], selected_feature_names
        elif method == 'lasso':
            lasso = LassoCV(cv=5, random_state=42)
            lasso.fit(features, y)
            mask = lasso.coef_ != 0
            selected_indices = np.where(mask)[0]
            selected_feature_names = [feature_names[i] for i in selected_indices]
            # Save indices of selected features in a file "selected_features.txt" in 'models' folder
            with open('models/selected_features.txt', 'w') as f:
                for index in selected_indices:
                    f.write("%s\n" % index)
            
            if verbose_features_selected :
                print(f"Selected {np.sum(mask)} features out of {features.shape[1]}")
                print(f"Selected features (lasso): {selected_feature_names}")
                
            return features[:, selected_indices], selected_feature_names
        elif method == 'mRMR':
            # Convert numpy array to pandas DataFrame if needed
            if not isinstance(features, pd.DataFrame):
                features_df = pd.DataFrame(features, columns=feature_names)
            else:
                features_df = features.copy()
            
            # Initialize MRMR selector
            # method : Random forest
            #mrmr_selector = MRMR(method="RFCQ",max_features=None, scoring="roc_auc",param_grid = {"n_estimators": [5, 15, 30], "max_depth":[1,2,3,4]},cv=3,regression=False, random_state=42)
            mrmr_selector = None
            # Fit and transform the data
            mrmr_selector.fit(features_df, y)

            # plot the relevance
            if verbose_features_selected:
                pd.Series(mrmr_selector.relevance_, index=mrmr_selector.variables_).sort_values(
                    ascending=False).plot.bar(figsize=(15, 6))
                plt.title("Relevance")
                plt.show()
            
            # transform the data to keep only the selected features 
            features_df = mrmr_selector.transform(features_df) 

            # Get indices of selected features
            selected_indices = mrmr_selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            # Save selected feature indices
            with open('models/selected_features.txt', 'w') as f:
                counter = 0
                for item in selected_indices:
                    if item == True:
                        # write the indices of the selected features
                        f.write("%s\n" % counter)
                    counter += 1
            
            # Print selected features if verbose mode is on
            if verbose_features_selected:
                print(f"Selected features (mRMR): {selected_feature_names}")
            
            # Return selected features and their indices
            return features[:, selected_indices], selected_feature_names
        elif method == 'elasticnet':
            # ElasticNet feature selection
            elasticnet = ElasticNetCV(cv=5, random_state=42, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
            elasticnet.fit(features, y)
            mask = elasticnet.coef_ != 0
            selected_indices = np.where(mask)[0]
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            # Save indices of selected features in a file "selected_features.txt" in 'models' folder
            with open('models/selected_features.txt', 'w') as f:
                for index in selected_indices:
                    f.write("%s\n" % index)
            
            if verbose_features_selected:
                print(f"Selected {np.sum(mask)} features out of {features.shape[1]}")
                print(f"Selected features (elasticnet): {selected_feature_names}")
            return features[:, mask], selected_feature_names 

    # Utils functions
    def _create_synapse_mask(self, original_image, coords_synapses, disk_radius=5):
        """
        Create a mask for synapses in the image.
        
        Args:
            original_image (np.ndarray): The original image
            coords_synapses (list): List of coordinates for synapses
            disk_radius (int, optional): Radius of the disk for each synapse. Defaults to 5.
        
        Returns:
            np.ndarray: The masked image with only synapses visible
        """
        mask_synapse = np.zeros_like(original_image)
        height, width = original_image.shape
        
        for coord in coords_synapses:
            rr, cc = draw.disk(coord, disk_radius)
            # Filter coordinates within image boundaries
            valid_coords = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            rr = rr[valid_coords]
            cc = cc[valid_coords]
            # Update mask with valid coordinates
            mask_synapse[rr, cc] = 1
        
        return mask_synapse

    def _watershed_segmentation(coordinates, image, mask):
        """Helper function for watershed segmentation"""
        markers = np.zeros_like(image, dtype=np.int32)
        for i, (x, y) in enumerate(coordinates, 1):
            markers[int(x), int(y)] = i + 100
        return ski.segmentation.watershed(-image, connectivity=1, markers=markers, mask=mask)

    def _find_additional_synapses(region, image, rough_segmented):
        """Helper function to find additional synapses in a region"""
        minr, minc, maxr, maxc = region.bbox
        mask = (rough_segmented[minr:maxr, minc:maxc] == region.label)
        
        # Calcul de l'intensité moyenne des bordures
        boundary = ski.morphology.dilation(mask, ski.morphology.disk(1)) ^ mask
        
        # Si aucun pixel de bord n'est trouvé, on ne peut pas calculer l'intensité moyenne
        if not image[minr:maxr, minc:maxc][boundary].size:
            return []
            
        # Calcul de l'intensité moyenne des pixels de bord
        # Cette valeur servira de référence pour détecter les synapses
        mean_boundary = np.mean(image[minr:maxr, minc:maxc][boundary])
        
        # Création de la fenêtre d'analyse en remplaçant l'arrière-plan par l'intensité moyenne des bords
        window = image[minr:maxr, minc:maxc].copy()
        window[~mask] = mean_boundary
        
        # Détection des maxima locaux
        local_maxima = ski.feature.peak_local_max(window, min_distance=2, exclude_border=False)
        
        # Filtrage des maxima : ne garde que ceux plus intenses que la moyenne des bords
        synapse_centers = [(x + minr, y + minc) 
                            for x, y in local_maxima 
                            if window[x, y] > mean_boundary]
        
        return synapse_centers

    def _get_regions_of_interest(self, coord, image_original, binary_mask):
        """
        Get regions of interest from the image using watershed segmentation.
        
        Args:
            coord: List of coordinates of synapse centers
            image_original: Original image
            binary_mask: Binary mask of the image
            
        Returns:
            Tuple of (region properties, segmented image)
        """
        
        # Première passe de segmentation
        rough_segmented = self._watershed_segmentation(coord, image_original, binary_mask)
        
        # Blurring the image
        image_smooth = ski.filters.gaussian(image_original, sigma=0.5)
        
        # Recherche de synapses supplémentaires
        additional_synapses = []

        for region in ski.measure.regionprops(rough_segmented, intensity_image=image_smooth):
            new_centers = self._find_additional_synapses(region, image_smooth, rough_segmented)
            additional_synapses.extend(new_centers)
        
        # Mise à jour des coordonnées et suppression des doublons
        coord = list(set(coord + additional_synapses))

        
        # Segmentation finale
        final_segmented = self._watershed_segmentation(coord, image_smooth, binary_mask)
        refined_segmented = np.zeros_like(final_segmented)
        
        # Raffinement des régions
        for region in ski.measure.regionprops(final_segmented, intensity_image=image_smooth):
            minr, minc, maxr, maxc = region.bbox
            mask = (final_segmented[minr:maxr, minc:maxc] == region.label)
            
            # Compute mean boundary intensity
            boundary = ski.morphology.dilation(mask, ski.morphology.disk(1)) ^ mask  # Find boundary pixels
            if not image_smooth[minr:maxr, minc:maxc][boundary].size:
                mean_boundary = 0
            else:
                mean_boundary = np.mean(image_smooth[minr:maxr, minc:maxc][boundary])
            
            
            # Construct new window
            new_window = image_smooth[minr:maxr, minc:maxc].copy()
            new_window[~mask] = mean_boundary  # Replace background with mean boundary intensity

            # Step 5: Apply K-means
            I = new_window / new_window.max()  # Normalize intensities
            B = mask.astype(float) * 0  # Binary weight
            features = np.column_stack((I.flatten(), B.flatten()))  # 2D feature space
            
            # Show feature space
            #IMAGE_DIAPO
            """plt.figure(figsize=(8, 6))
            plt.scatter(features[:, 0], features[:, 1], c=features[:, 0], cmap='viridis')
            plt.title(f"Feature space for region {region.label}")
            plt.xlabel("Intensity")
            plt.ylabel("Binary weight")
            plt.show()"""
            
            if features.shape[0] < 2: # Not enough features for region
                labels = np.zeros_like(features)
                continue
            
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            labels = kmeans.fit_predict(features)
        
            # Step 6: Determine foreground and update segmentation
            refined_region = labels.reshape(mask.shape)
            foreground_label = np.argmax([np.mean(I[refined_region == 0]), np.mean(I[refined_region == 1])])
            foreground_mask = (refined_region == foreground_label)
            
            
            # Keep only the largest connected component in the foreground        
            labeled_fg = ski.measure.label(foreground_mask, connectivity=1)
            regions = ski.measure.regionprops(labeled_fg)
            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                largest_component_mask = (labeled_fg == largest_region.label)
                filled_component_mask = scipy.ndimage.binary_fill_holes(largest_component_mask, structure=np.array([[0,1,0],
                                                                                                                    [1,1,1],
                                                                                                                    [0,1,0]])) 
            else:
                filled_component_mask = np.zeros_like(foreground_mask)
            
            # Update refined mask
            refined_mask = np.zeros_like(mask, dtype=np.int32)
            refined_mask[filled_component_mask] = region.label
            refined_segmented[minr:maxr, minc:maxc][refined_mask > 0] = region.label
            
            # IMAGE_DIAPO
            """# Create a figure with a specified size for the combined image
            fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # 1 row, 3 columns
            # Visualize region (first subplot)
            axes[0].imshow(mask, cmap='gray')
            axes[0].set_title(f"Region {region.label}")
            axes[0].axis('off') # Turn off axis labels and ticks
            # Visualize new window (second subplot)
            axes[1].imshow(new_window, cmap='gray')
            axes[1].set_title(f"New window for region {region.label} with mean boundary intensity for background")
            axes[1].axis('off')
            # Visualize refined region (third subplot)
            axes[2].imshow(refined_mask, cmap='gray')
            axes[2].set_title(f"Refined region {region.label}")
            axes[2].axis('off')
            # Adjust layout to prevent overlapping titles
            plt.tight_layout()
            # Show the combined image
            plt.show()"""
        
        # Calcul des propriétés finales des régions
        region_props = ski.measure.regionprops(refined_segmented, intensity_image=image_original)
        
        return region_props, refined_segmented

    def _extract_texture_features(self, roi_intensity: np.ndarray, roi_mask: np.ndarray, prop, median_width) -> np.ndarray:
        """
        Extract texture features from region of interest of an image.

        Args:
            roi_intensity: numpy array of the intensity of the region of interest
            roi_mask: numpy array of the mask of the region of interest
            prop: region properties of the region of interest
            median_width: median width of the worm, used for normalization

        Returns:
            numpy array of the texture features
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = []
        
            try: # First Order Statistics
                fos_feat, _ = pyfeats.fos(roi_intensity, roi_mask)
                fos_feat = np.nan_to_num(fos_feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(fos_feat)
            except Exception as e:
                features.extend([0.0] * 16)

            try: # Neighborhood Gray Tone Difference Matrix
                ngtdm_feat, _ = pyfeats.ngtdm_features(roi_intensity, roi_mask, d=1)
                ngtdm_feat = np.nan_to_num(ngtdm_feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(ngtdm_feat)
            except Exception as e:
                features.extend([0.0] * 5)
                
            try: # Gray Level Difference Statistics
                glds_feat, _ = pyfeats.glds_features(roi_intensity, roi_mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
                glds_feat = np.nan_to_num(glds_feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(glds_feat)
            except Exception as e:
                features.extend([0.0] * 5)
            
            """try: # Statistical Feature Matrix
                height, width = roi_intensity.shape
                min_dim = min(height, width)
                if min_dim >= 4:
                    Lr = Lc = min(4, min_dim // 2)  # Utiliser au maximum 4 ou la moitié de la plus petite dimension
                    sfm_feat, _ = pyfeats.sfm_features(roi_intensity, roi_mask, Lr=Lr, Lc=Lc)
                    sfm_feat = np.nan_to_num(sfm_feat, nan=0.0, posinf=0.0, neginf=0.0)
                    features.extend(sfm_feat)
                else:
                    features.extend([0.0] * 4)  # SFM retourne 4 caractéristiques
            except Exception as e:
                features.extend([0.0] * 4)"""
            
            try: # Statistical Feature Matrix
                sfm_feat, _ = pyfeats.sfm_features(roi_intensity, roi_mask, Lr=4, Lc=4)
                sfm_feat = np.nan_to_num(sfm_feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(sfm_feat)
            except Exception as e:
                features.extend([0.0] * 4)
            
            try: # Gray Level Run Length Matrix
                glrlm_feat, _ = pyfeats.glrlm_features(roi_intensity, roi_mask, Ng=256)
                glrlm_feat = np.nan_to_num(glrlm_feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(glrlm_feat)
            except Exception as e:
                features.extend([0.0] * 11)

            try: # Fourier Power Spectrum
                fps_feat, _ = pyfeats.fps(roi_intensity, roi_mask)
                fps_feat = np.nan_to_num(fps_feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(fps_feat)
            except Exception as e:
                features.extend([0.0] * 2)

            try: # Gray Level Size Zone Matrix
                glszm_feat, _ = pyfeats.glszm_features(roi_intensity, roi_mask)
                glszm_feat = np.nan_to_num(glszm_feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(glszm_feat)
            except Exception as e:
                features.extend([0.0] * 14)

            try: # Shape Parameters
                perimeter = prop.perimeter
                shape_feat, _ = pyfeats.shape_parameters(roi_intensity, roi_mask, perimeter, pixels_per_mm2=1)
                shape_feat = np.nan_to_num(shape_feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(shape_feat/median_width if median_width > 0 else shape_feat)
            except Exception as e:
                features.extend([0.0] * 5)
        
        return np.array(features)
    
    def _extract_spatial_features(self, graph: nx.Graph, component_props) -> np.ndarray:
        """
        Extract spatial features from the graph, including both node distances and edge statistics.
        
        Args:
            graph: NetworkX graph containing node positions and edges
            component_props: List of region properties for the components in the graph
            
        Returns:
            Array of spatial features
        """
        # Get node positions from the graph
        centroid_positions = np.array([prop.centroid for prop in component_props])
        
        features = []
        
        if len(centroid_positions) > 1:
            # Compute distance matrix between all points
            dist_matrix = distance_matrix(centroid_positions, centroid_positions)
            np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
            nearest_neighbor_dists = np.min(dist_matrix, axis=1)
            
            # Calculate statistics of distances
            features.extend([
                np.mean(nearest_neighbor_dists),  
                np.std(nearest_neighbor_dists),   
                np.min(nearest_neighbor_dists),   
                np.max(nearest_neighbor_dists)    
            ])
            
            # Compute Ripley's K function
            max_dist = np.max(nearest_neighbor_dists) * 2
            radii = np.linspace(0, max_dist, 5)
            
            for r in radii:
                if r > 0:
                    count = np.sum(dist_matrix < r, axis=1)
                    k_r = np.mean(count) / (len(centroid_positions) - 1)
                    features.append(k_r)
                else:
                    features.append(0.0)
                    
            # Add edge length statistics
            try:
                # Compute the length of all edges in graph
                edge_lengths = []
                for u, v in graph.edges():
                    # Get centroids from node attributes
                    u_pos = np.array(graph.nodes[u]['centroid'])
                    v_pos = np.array(graph.nodes[v]['centroid'])
                    length = np.linalg.norm(u_pos - v_pos)
                    edge_lengths.append(length)
                
                edge_lengths = np.array(edge_lengths)
                
                if len(edge_lengths) > 0:
                    # Compute statistics
                    mean_distance = np.mean(edge_lengths)
                    max_distance = np.max(edge_lengths)
                    median_distance = np.median(edge_lengths)
                    min_distance = np.min(edge_lengths)
                    number_edges_sup_mean = len(edge_lengths[edge_lengths > 3*median_distance])/len(edge_lengths)
                    
                    # Add features to vector
                    features.extend([
                        mean_distance,
                        max_distance,
                        median_distance,
                        min_distance,
                        number_edges_sup_mean
                    ])
                else:
                    features.extend([0.0] * 5)
            except Exception as e:
                # In case of error, add zeros for edge features
                features.extend([0.0] * 5)
                print(f"Error processing edges: {e}")
                
        else:
            # Add zeros for all features if too few nodes
            features.extend([0.0] * (4 + 5 + 5))  # 4 for distance stats + 5 for Ripley's K + 5 for edge features
        
        return np.array(features)
    
    def _aggregate_region_features(self, region_features: List) -> List[float]:
        """
        Agrège les caractéristiques de toutes les régions en calculant des statistiques.
        
        Args:
            region_features: Liste des caractéristiques pour chaque région
            
        Returns:
            Liste des caractéristiques agrégées
        """
        aggregated_features = []
        
        if region_features:
            # Convertir en array numpy pour faciliter les calculs
            region_features = np.array(region_features)
            
            # Calculer les statistiques
            mean_features = np.mean(region_features, axis=0)
            std_features = np.std(region_features, axis=0)
            min_features = np.min(region_features, axis=0)
            max_features = np.max(region_features, axis=0)
            
            # Ajouter toutes les statistiques aux caractéristiques
            aggregated_features.extend(mean_features)
            aggregated_features.extend(std_features)
            
            # Pour certaines caractéristiques importantes (FOS et NGTDM), ajouter aussi min et max
            aggregated_features.extend(min_features[:21])
            aggregated_features.extend(max_features[:21])
        else:
            # Si pas de régions, ajouter des zéros
            n_features = 62
            aggregated_features.extend([0.0] * n_features * 2)  # Pour mean et std
            aggregated_features.extend([0.0] * 21 * 2)  # Pour min et max des FOS et NGTDM
            
        return aggregated_features

    def _process_features(self, y, indices_coiled):
        features = np.zeros((len(self.features), len(self.feature_names)))
        for i in range(len(self.features)):
            if i not in indices_coiled:
                features[i] = self.features[i]
            else:
                features[i] = np.zeros(len(self.feature_names))
        features_name = self.feature_names
        
        # Detect indice of elements in features which contain only 0s -> Coil worm
        indices_row = np.where(np.all(features == 0, axis=1))[0]
        # Detect features with all 0s
        indices_column = np.where(np.all(features == 0, axis=0))[0]
        
        # Remove these elements 
        features = np.delete(features, indices_row, axis=0)
        y = np.delete(y, indices_row, axis=0)
        features = np.delete(features, indices_column, axis=1)
        features_name = np.delete(features_name, indices_column, axis=0)
        
        # Convert labels to numeric
        unique_labels = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])
        
        return features, features_name, y
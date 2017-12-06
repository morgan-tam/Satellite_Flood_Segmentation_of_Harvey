# Semi-Supervised Flood Segmentation of Satellite Images of Houston After Harvey. 

**Problem:** 
During this past summer, there’s been an abnormal amount of natural disasters in North and South America. The frequency of hydrological catastrophes has been on the rise since the 80s. For a final project at Metis, I want to create a method to detect flood automatically on streaming satellite data, so authorities and first responders can prioritize relief efforts.

![Natural Disaster Trends from Economist](Images/Economist_Trend.png?raw=true "Title")
Fig. 1 Natural Disaster Trends from The Economist.

![Personal photo of St. Marteen after Hurrican Irma](Images/St.Marteen.png?raw=true "Title")
Fig. 2 A picture taken by my friend in St. Marteen right after Irma.


**Data:** 
DigitalGlobe recently started their Open Data Program, generously sharing their satellite data of locations before and after a natural disaster. My particular data is a three band (R,G,B) high resolution aerial image before and after Harvey in Houston. The ground truth is from the MDA shapefile that used Radarsat to detect flood (https://www.digitalglobe.com/opendata).
![Houston](Images/Entire_Data.png?raw=true "Title")
Fig. 3 Satellite Data of Houston Post-Harvey.


**Challenges:** 
1. I had a time constraint of two weeks to present instead of four due to the fact that the data came out in the middle of the project timeline. 

2. The data was a decent size of 180 GB due to the very high resolution of satellite images of the whole city of Houston.

3. Geospatial data is challenging to handle and require many tools for analysis (ie QGIS, GDAL, Rasterio, Geopandas, Shapely). There is also an added difficulty of matching coordinates (earth is round and there are many ways to convert to into 2D).

4. The provided ground truth from MDA's Radarsat in the form of shapefiles is very low resolution and has a mixture of false negatives and false positives. 

5. Because of the incomplete ground truth segmentation, there is no good metric to judge the models' performance. I used Jaccard Similarity, but I still have to take it as a grain of salt.
![MDA Shapefile](Images/MDA_mask.png?raw=true "Title")
Fig. 4 Incomplete Radarsat Ground Truth.

6. There are many different types of water bodies, anad it would not be trivial in distinguishing flood from natural/man made bodies of water. There is also a challenge of separating houses from non-flood homes, as the roof may be naturally brown.
![Flood segmentation in residential areas](Images/Challenge-houses.png?raw=true "Title")
Fig. 5 Flood segmentation in residential areas. 
 

**My Approach:**
1. Create Initial Input Data
I used multiprocessing to split up large images (1GB each) into many manageable tiles and generate ground truth masks from the given Radarsat shapefiles for corresponding image tiles.


2. Train U-Net Model on Radarsat Data
For my baseline model, I chose a U-Net for segmentation. U-Net is like CNN that encode & decodes but can skip connections that are on the same “level". It is good for prototyping, as it does not require 
learning to perform well (https://arxiv.org/abs/1505.04597).
Because of the large dataset, I had to use a generator to load the large data in mini-batches.
![U-Net Architecture](Images/UNet_Architecture.png?raw=true "Title")
Fig. 6 U-Net Architecture

My result for the original data set is **49% in Jaccard Similarity**. Interestingly, the predictions would occasionally perform better than the original Radarsat ground truth.
![Original_U-net_result](Images/Original_U-net_result.png?raw=true "Title")
Fig. 7 U-net result of Original Data
 

3. Train ResNet
For my next model, I chose Residual Net. ResNet is a state-of-the-art CNN that is excellent at object detection and image segmentation (https://arxiv.org/abs/1512.03385). I used pre-trained weights from the ImageNet competition.
![Resnet](Images/Resnet.png?raw=true "Title")
Fig. 8 Resnet Identity block.

My result for this model is **77% in Jaccard Similarity** which is a huge improvement. However, it started behaving exactly like the radar, learning even its flaws. This is overfitting in the classic sense.
![Resnet_result](Images/Resnet_result.png?raw=true "Title")
Fig. 9 Resnet result of Original Data
 

4a. K-means Exploration
Because the models' performance is bottlenecked by the incomplete Radarsat ground truth, I decided to go with an unsupervised approach. Here, I explored using K-means clustering, which is a popular clustering algorithm for general data. Once the users choose how many clusters they want, K-means will find the best centroid for each cluster, including images. However, choosing the right K is a challenge in itself, espeically for varying images.

I Initialized with an average pixel center instead of finding random centroids or even using K++. For each observation, I iterate between 4-8 clusters choosing one that overlaps the most with the original ground truth. I Experimented with different intersections and union with U-Net predictions and radarsat ground truth. The masks look more promising but has high chance of false positive (which might be okay for prototype).


4b. Create K-means Clustered Input Data
Here I recreate new masks using K-means, but because I needed all the cores for the clustering model, I didn't use multiprocessing for creating the data.

I end up choosing intersecting with the prediction and union with Radarsat ground truth. 	This looks alright, but it still needs some Gaussian smoothing.
![K-means masks](Images/K-means_mask.png?raw=true "Title")
Fig. 10 New K-means Ground Truth Mask


4c. Train U-Net Model on K-means Clustered Data
This time, U-Net results were much better, but it was still hard to use any metric to measure, had to manually visualize.
![K-means Result](Images/K-means_Result.png?raw=true "Title")
Fig. 11 K-means Result
  
  
5a. DBSCAN Exploration 
Next, I went with DBSCAN. DBSCAN is more appropriate for this type of data and unlike K-means, I do not have to specify how many clusters I need. It is not biased towards any cluster and becasue the images vary signficantly, this algorithm can perform better.	I do have to choose two parameters, radisu of clusters and minimum of points in each cluster. I chose a few values using the KNN distance plot and trial and error.
![KNN Distance Plot](Images/KNN_Distance.png?raw=true "Title")
Fig. 12 KNN Distance Plot

I also created filter threshold to remove clusters that are too similar to vegetation (green), clouds (white), and buildings (grey).


5b. Create DBSCAN Clustered Input Data 
The implementation is same as step 4b.

![DBSCAN Ground Truth Mask](Images/DBSCAN_Exploration.png?raw=true "Title")
Fig. 13 DBSCAN Ground Truth Mask


5c. Train U-Net Model on DBSCAN Clustered Data 
The Jaccard Similarity is very low, **53%**, however the results don't look too bad.

![DBSCAN U-Net Result](Images/DBSCAN_UNET.png?raw=true "Title")
Fig. 14 DDBSCAN U-Net Result
	 
   
6. Manually Selecting Training Data 
Here comes the semi-supervised part. I needed to quickly select which images are good, so I uploaded the path to all my data to Redis server, so multiple users can pull information without any redundancy. I then prompt for choice, whether the mask is good, bad, or can be replaced by the U-Net prediction. Users can remotely go to copies of the jupyter notebook and organize data based on its quality. I remove the bad data and replace with better ones.

Out of 1240 images that I went through, 15% of masks were good, 30% were bad, 55% could be replaced by U-Net predictions. This means there can be a lot more improvements for the DBSCAN process, but surprisingly, the U-Net predictions were able to pick off from the mask and do a better job. This hints that predictions can be used further in a sem-supervised approach.


7. Train U-Net on Final Data
Now I'm going to feed back the model handpicked images, most were predictions from the previous model.
The results I got was **80% in Jaccard similarity*, but there were only 50 validation set so that might be why the metric was so much higher. Visually, it does loom better than the final mask. It can even perform well when image is under shadow of a cloud.
![Final Test Result](Images/Final_Validation_2.png?raw=true "Title")
Fig. 15 Final Test Result

Let's take a look at a few examples how this model performs against the original ground truth mask and DBSCAN mask.
![Model Prediction vs. Old Masks](Images/Comparison_with_old_4.png?raw=true "Title")
Fig. 16 Model Prediction vs. Old Ground Truth
	
It performed fairly well against the old ground truth, but this process still need improvements. It doesn’t do well with clouds and sometimes houses.


**Next steps:**
For next time, I want to try
-Spectral clustering

-Stitching images together

-Manually selecting more data

-Change detection with geolocation metadata saved

-Streaming live data


### Summary Exploration 1

used file: MAG2P_order_parameters-2025-5-5-19:38:49.pickle
date: 8th of May, 2025
data availability: the data can be currently found in: rigid_magnetic/results/ML/Exploration_1

We conducted a first exploration of our feature set with ML algorithms. 
The feature set contains a set of order parameters produced with mag2patch_nbonds.py. Beside mean and std values it also contains the probability distribution of 
mutual magnetic moment distributions. 

df.columns

Index([                'file_id',                  'lambda',
                         'shift',              'mean_bonds',
                     'std_bonds',  'mean_second_neighbours',
         'std_second_neighbours',               'mean_size',
                      'std_size',                 'largest',
       'mean_radius_of_gyration',  'std_radius_of_gyration',
                             0.0,       0.12566370614359174,
             0.25132741228718347,        0.3769911184307752,
              0.5026548245743669,        0.6283185307179586,
              0.7539822368615504,        0.8796459430051422,
              1.0053096491487339,        1.1309733552923256,
              1.2566370614359172,        1.3823007675795091,
              1.5079644737231008,        1.6336281798666925,
              1.7592918860102844,         1.884955592153876,
              2.0106192982974678,        2.1362830044410597,
               2.261946710584651,         2.387610416728243,
              2.5132741228718345,        2.6389378290154264,
              2.7646015351590183,        2.8902652413026098,
              3.0159289474462017,        3.1415926535897936],
      dtype='object')

For first explorations of the data set we used **TSNE** and PCA and colored the output according to $\lambda$ and shift. 
The TSNE already show that point similar in $\lambda$ and shift, with a wide band over all shifts that contains un-assembled (liquid) structures and smaller clusters for the high $\lambda$ and high shifts where chains, closepacked and porous structures are expected. 
**UMAP** seems to give a similar clustering as TSNE. 

Equally the **PCA** also shows a lot of structure, although the cluster separation is clear. The points ind 2D and 3D plots present more as meandering single structure where points with similar shifts/$\lambda$ are in a similar position but not clustered togheter. 
Note that it is necessary to scale all variables to have a range between 0 and 1, otherwise the PCA fails to grasp any structure. 
The loadings heatmap shows that for each PC different components contribute and the strongly negative and postive values indicate/underline the graphical observation that the data contains a lot of distinguishing features. The explained variance plots that a surprising amount of feature is necessary to get over 90% explained variance (18). We note that we briefly explored what happens if we only used known maximas from the mutual orientations of the magnetic moments, and in this case only six components are needed to push the variance plot over 90%. We wonder why that is the case. Is it because there is so much more added information or is it because the historgram als adds noise that has to be compensated with more components? 

To test the performace of the dimensionality reductiion through TSNE and UMAP as well as high dimensional clustering we plotted a state diagram across shift and $\lambda$ to see whether these automatic structure detections match up with our expectations fueled by visual inspection of the simulation results. 

From a first glance, the direct high dimensional clustering (HDBSCAN) seems to fare best in caputuring some of the regions: chains (red), liquid (blue), the different networked structure (orange, yellow, and lilac), the kagome (light lilac). However, it seems to fail to detect the porous disordered structures between s=0.45 and s=0.55. It might have to do with the fact that these structures, due to their porosity and disorder, have a RDF that has no first minimum, instead there is a plateau that signifies that there a lot of possibilites for first to second neighbur alignment. In contrast the other porous clusters - the Kagome like structures have a sharp minimum at around 1.5. (see accompanying plot below). 

The state diagram derived by the clustering of the 2D UMAP, seems to somewhat the liquid line for all shifts, but seems rather disorganized for everything else. There are too many clusters than seems reasonable from visual inspection. 

The state diagram derived by TSNE seems to see stiff chains, the kaogme and some networked strucuters, but fails to distinguish liquid from rings/chains for small shifts.

[200~### Next steps 

There is a couple of avenues to explore when going forward: 

#### Changing the cutoff 
First off, inspecting the RDF again for some structures, made me reconsider the cutoff range: For this exploration it was 1.3, but it actually seems to be closer to 1.5 for the Kagome structures and for the porous disordered structures at s=0.5, there is not even a second minimum, making the choice of the cutoff difficult, as it is usally chosen to be the first minimum of the RDF. A first step will be at least to extend it to 1.5. It is a good  thing, that I already have a quantity called second_neighbours, that captures everything between 1.3 and 2.0 -- unsurprisingly this parameter lights up for the region between s= 0.45 and 0.55 where we expect the porous disordered structures. 

#### Using all data points and not averages per state point 
It might be interesting to try this, as for the methods given, averaging is strictly speaking not necessary. It the last step, when generating the state diagram, I will have to bin the results, to get meaningful state point values. 

#### Using known and apriori unknown maxima in the mutual moment orientations 
I have already prepared a routine that allows to find all maxima and then evaluates the histograms for all data points for these maxima positions. This is a way to 
agnostically handle moment orientations: we do not presume we know the maxima beforehand. 
I have however also used the known maxima: 0,60,90,120 and 180 degrees. While it limits the information, it may reduce the noise due to high dimensionality. 

#### Using the full radius of gyration function
Similarly as with the orientation histograms I intend to use the full radius of gyration function, that is dependent on the cluster size. 

#### Using autoencoders and compare them with the PCA results 
Autoencoder latent spaces are thought to be equivalent to PCA biplots. We have the opportunity to inspect that. 

#### Test the robustness of clusteirng methods for state diagram. 
To understand if clustering of original data/reduced data is relyable accross different clustering hyper parameters, we have to explore these parameter spaces. 

#### Testing the performance of fancier tools 
Find out if there is more up-to-date clustering/unsupervised learning methods and try them out on the data set 

### Testing with raw positions 
Similarily as with the autoencoder project: can we learn on raw positions?  

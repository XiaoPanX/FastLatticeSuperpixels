# FastLatticeSuperpixels
This is the code of our paper with title "Fast Generation of Superpixels With Lattice Topology" published on TIP 2022.
In this paper, we propose a new superpixel generation method which can generate superpixels with lattice topology, i.e., the generated superpixels have regular lattice topology just like pixels on images. We achieve this point by moving the boundary pixels (or blocks) with constraints to optimize the boundaries of superpixels, as shown in the following figure:

![image](https://github.com/XiaoPanX/FastLatticeSuperpixels/blob/compute_laticce_spixels/pipeline.png)
The initial superpixels are shown in (a). We adjust the boundaries of superpixels from coarse to fine by moving the multilevel blocks as shown in (b)-(d). We refine the boundaries by adjusting the boundary pixels to get the final superpixels as shown in (e)

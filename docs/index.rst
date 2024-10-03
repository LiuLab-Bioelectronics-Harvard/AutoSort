FuseMap
=====================================


Spatial integration and mapping with universal gene, cell, and tissue embeddings.
-----------------------------------------------------------------------------------
FuseMap is a deep-learning framework for spatial transcriptomics 
that (1) bridges single-cell or single-spot gene expression within spatial contexts 
and (2) consolidates various gene panels across technologies, organs, and species.


.. image:: _static/framework.png
    :width: 100%
    :align: center

------------------------------------------

Contents
------------------------------------------
.. toctree::
   :maxdepth: 2

    Overview <self>
    Installation <install>
    Tutorials <tutorials>
    API Reference <api>
    About <about>

------------------------------------------

Quick start
------------------------------------------

Spatial integration
^^^^^^^^^^^^^^^^^^^^^
:mod:`fusemap.spatial_integrate` provides tools to integrate spatial transcriptomics data.
Input data can be from any spatial transcriptomics technology, such as Visium, Slide-seq, or MERFISH.
Output data can be used for downstream analysis, such as clustering, cell type identification, or spatial gene expression analysis.


Spatial mapping
^^^^^^^^^^^^^^^^^^^
:mod:`fusemap.spatial_map` provides tools to map spatial transcriptomics data to a universal gene, cell, and tissue embedding space.


Step-by-step guide 
^^^^^^^^^^^^^^^^^^^
Check out our detailed `tutorials <tutorials.html>`_ on how to use FuseMap for spatial integration and mapping.


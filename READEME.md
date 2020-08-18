The source code is an implementation of our method described in the paper "Guang-Hui Liu, Isabelle Bichindaritz*, Christopher Bartlett, and Boseon Byeon. Survival Analysis of Breast Cancer Utilizing Integrated Features with Ordinal Cox Model and Auxiliary Loss". 

Because github limit that the uploaded file size cannot exceed 100Mb, two original mRNA and methylation datasets fail to upload in github repository. So the whole source code for the proposed method and the datasets have been put online available at https://pan.baidu.com/s/1sp356aQePGDj8vz2b89rpA (extracted code is: LGH0) for free academic use. You will be appreciated to download from this address.

--Dependence

Before running this code, you need install R and python. In our experiment, R3.62 and python 3.6 or more advanced version are tested. This code is tested in WIN7/win10 64 Bit. It should be able to run in other Linux or Windows systems.

--How to run

   step1. In R, in folder: \LSTM-COX-CODE\mRNA data\, first,run notebook-mRNAseq_RPKM.r; then, run brca-wgcna_mRNAseq_RPKM.r. Similarly, in folder: \LSTM-COX-CODE\methylation data\, first,run notebook-merge-datasets-methylation.r; then run brca-methylation_wgcna.r. If the code runs successfully, the extracted mRNA features and methylation features will be obtained,respectively. We combine mRNA and methylation features in sequence and obtain a 36-dimensional feature vector which will be viewed as integrated gene feature input.
   
   step2. After obtaining the specific feature representation in step1, copy the integrated features to fold \LSTM-COX-CODE\survival analysis\. In Python, run brca-keras_methylation_mRNA_WGCNA_ordinal_loss.py in fold \LSTM-COX-CODE\survival analysis\.
   
--Output

if the code runs successfully, the results will be placed in \LSTM-COX-CODE\survival analysis\c_indices.npy. 


Email:Guanghui.liu@oswego.edu.


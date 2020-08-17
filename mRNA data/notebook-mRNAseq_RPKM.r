setwd("C:/2019821topc/mRNA")

#install.packages("cghRA")

#install.packages("dplyr")

#install.packages("stringr")

#install.packages("ggplot2")

#install.packages("data.table")
#install.packages("readxl")

library(readxl)
library(cghRA)
library(dplyr)
library(stringr)
library(ggplot2)
library(data.table)

#tableGenes <- read_excel("BRCAMergedWAVE.xlsx")
tableGenes <- read.table("BRCA.mRNAseq_RPKM.txt", sep="\t",
                               header = F, fill = T, stringsAsFactors=FALSE,quote = "")
tableClin <- read.table("brca_tcga_clinical_data.tsv", sep="\t", 
                         header = T, fill = T, stringsAsFactors=FALSE)

tableGenesT <- transpose(tableGenes, fill=NA, ignore.empty=FALSE)

colnames(tableGenesT) = tableGenesT[1, ] # the first row will be the header 第一行提升作为头，
tableGenesT = tableGenesT[-1, ] #去掉第一行

tableGenesT22=tableGenesT$HYBRIDIZATION_R
#tableGenesT$Hybridization_REF<-substring(tableGenesT22,1,15)

tableClin22=tableClin$Sample.ID

tableClinGen <- merge(x=tableClin, y=tableGenesT, by.x="Sample.ID", by.y="HYBRIDIZATION_R")

write.table(tableClinGen, "tableClinGen_mRNAseq_RPKM.csv", sep=",", row.names=FALSE, col.names=TRUE)
# move back column headers into first row
tableClinGen2 <- rbind(colnames(tableClinGen), tableClinGen)

tableClinGenT <- transpose(tableClinGen2, fill=NA, ignore.empty=FALSE)

# cell #1 read data
#data1 <- read.table("BRCAMergedWAVE.csv", sep=",", 
            #header = F, fill = T, stringsAsFactors=FALSE)

# following code is when loading the merged dataset from a file
# first col is int, 2nd col is Des.GeneSymbol, 3rd col is Des.Platform, 4th row is Des.Description
# cols 5 and above are Data.TCGA.3C.AAAU.01 etc.

# data1 <- read.table("BRCAMerged.csv", sep=",", 
#                    header = F, fill = T, stringsAsFactors=FALSE)

# cell #2 111026 observations 2242 variables
#dim(data1)
#class(data1)

# cell #3
#summary(data1) # summary statistics

# cell #4 add sampleType 3778 observations 2242 variables
getSampleType <- function(x)
{ 
#  print(x)
#  print(class(x))
  return(
# use for WAVE    unlist(strsplit(x, "\\."))[4]
    unlist(strsplit(x, "\\-"))[4]
#    unlist(strsplit(x, "\\."))[5]
  )
}

# getSampleType("Data.TCGA.3C.AAAU.01")
getSampleType("TCGA-3C-AAAU-01")

data2 <- tableClinGenT
class(data2)
data3 <- data2[1, -(1:1)]
class(data3)
data4 <- lapply(as.character(data3), getSampleType)
data4 <- append(data4, c('sampleType'),0) 
data2 <- rbind(data2[1:3,],data4,data2[-(1:3),])

# cell #5 add type 
# C (primary cancer 01-05) N (normal 20-23) MN (matched normal 10-19) M (metastasis 06)
unique(data4[-(1:1)])
# extracts how many unique objects there are 

tab <- table(unlist(data4[-(1:1)]))
tab
# count how many of each type
#   01   06   10   11 
#  1093   7    0    0 

getClass <- function(x)
{ 
  n <- as.integer(x)
  print(n)
  # print(class(n))
  if (is.na(n)) {cl <- " "}
  else if ((n >= 01) & (n <= 05))
   {cl <- "C"}
  else 
    if (n == 06)
    {cl <- "M"}
    else 
       if ((n >= 10) & (n <= 19))
          {cl <- "MN"}
       else 
         if ((n >= 20) & (n <= 29))
         {cl = "N"}
         else {cl <- " "}
  return(cl)
}

data3 <- lapply(as.character(data4[-(1:1)]), getClass)
data5 <- append(data3, c('type'),0) 
data3 <- rbind(data2[1:4,],data5,data2[-(1:4),])

unique(data5[-(1:1)])
# extracts how many unique objects there are 

tab2 <- table(unlist(data5[-(1:1)]))
tab2

# count how many of each type
#   C     M   MN    
#  1093   7    0 

# cell #6 change gene variables to add gene name: ABI1 + NA -> ABI1_CNA rows 3725 until the last row
# no required clinical <= 145, genes >=146
data3[146, 1]


# cell #7 transpose, save
# add one row for stage as 1, 2, 3, 4
# 20586 rows and 1101 columns

getStage <- function(x)
{ 
  print(x)
  # print(class(n))
  if (is.na(x)) {cl <- "NA"}
  else if (startsWith(x, "Stage IV"))
  {cl <- "4"}
  else 
    if (startsWith(x, "Stage III"))
    {cl <- "3"}
  else 
    if (startsWith(x, "Stage II"))
    {cl <- "2"}
  else 
    if (startsWith(x, "Stage I"))
    {cl = "1"}
  else {cl <- "NA"}
  print(cl)
  return(cl)
}

data3[1, 1]
data3[8, 1]
data3[7, 1]
data3[9, 1]
data3[9, 2]
data3[146, 1]
data3[147, 1]
getStage(data3[9, 3])

datas <- data3[9, -(1:1)]
class(datas)
datastage <- lapply(as.character(datas), getStage)
datastage <- append(datastage, c('stage'),0) 
data5 <- rbind(data3[1:5,],datastage,data3[-(1:5),])

unique(datastage[-(1:1)])
# extracts how many unique objects there are 

tab3 <- table(unlist(datastage[-(1:1)]))
tab3

# count how many of each type
#   1     2   3       4     NA   
#  181   624  251     20    24     (total: 1100)


#data6 <- data5[ -c(1:3) ]
#data6[1,1] <- "sampleID"

data5[1,1]
data5[147,1]

# cell #8 select pertinent clinical and gene data, transpose, and save

options(max.print = 999999999)

data7 <- data5
data7$V1[147]
nrow(data7)

data8 <- data7[147:nrow(data7),]

#data7[grepl('_scaledExp', data7$V4), ]

data7 <- data5
data10 <- 
  data7 %>%  
  filter(
      data7$V1 %in% 
        c("Sample.ID", 
       "Patient.ID", 
       "sampleType", 
       "type",
       "stage",
       "Diagnosis.Age",
       "American.Joint.Committee.on.Cancer.Metastasis.Stage.Code",
       "Neoplasm.Disease.Lymph.Node.Stage.American.Joint.Committee.on.Cancer.Code",
       "Neoplasm.Disease.Stage.American.Joint.Committee.on.Cancer.Code",  
       "American.Joint.Committee.on.Cancer.Tumor.Stage.Code",
       "Cancer.Type.Detailed",
       "Birth.from.Initial.Pathologic.Diagnosis.Date",
       "Days.to.Sample.Collection.",
       "Death.from.Initial.Pathologic.Diagnosis.Date",
       "Last.Alive.Less.Initial.Pathologic.Diagnosis.Date.Calculated.Day.Value",
       "Days.to.Last.Followup",
       "Disease.Free..Months.",
       "Disease.Free.Status",
       "ER.Status.By.IHC",
       "ER.Status.IHC.Percent.Positive",
       "Ethnicity.Category",
       "Fraction.Genome.Altered",
       "HER2.fish.status",
       "HER2.ihc.percent.positive",
       "HER2.ihc.score",
       "Neoplasm.Histologic.Type.Name",
       "Prior.Cancer.Diagnosis.Occurence",
       "IHC.HER2",
       "Year.Cancer.Initial.Diagnosis",
       "Menopause.Status",
       "Patient.Metastatic.Sites",
       "Overall.Survival..Months.",
       "Overall.Survival.Status",
       "Patient.Primary.Tumor.Site",
       "PR.status.by.ihc",
       "PR.status.ihc.percent.positive",
       "Race.Category",
       "Sample.Type",
       "Sex",
       "Person.Neoplasm.Status"
    ))

data7$V1[20:40]
data10$V1

data11 <- rbind(data10,  data8)

#data12[] <- t(data11)
data12 <- transpose(data11, fill=NA, ignore.empty=FALSE)

class(data12)


# take only primary tumor samples
data14 <- 
  data12 %>%  
  filter( V4 ==  "C" | V4 ==  "N" | V4 == "type" )


write.table(data14, "BRCA_mRNAseq_RPKM.csv", sep=",", row.names=FALSE, col.names=FALSE)

#write.table(data14, "BRCAMerged2.csv", sep=",", row.names=FALSE, col.names=FALSE)



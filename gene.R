# This script downloads the gene expression data for use with `gene.py`.


# Install the required packages.
for (tmp in c("BiocManager", "httr")) {

    if(!tmp %in% rownames(installed.packages())) install.packages(
        pkgs = "BiocManager",
        repos = "https://cloud.r-project.org",
        dependencies = T
    )

}


for (tmp in c("readxl", "RTCGA", "RTCGA.mRNA")) {
  
  if(!tmp %in% rownames(installed.packages())) {
    BiocManager::install(pkgs = tmp, dependencies = T, update = F)
  }

}


# The following function to read and transpace a CSV file is taken from
# https://gist.github.com/jonocarroll/b17ce021b0637a31f584ed08a1fbe733#file-read-tscv-r-L1-L20.
## Based on
## https://stackoverflow.com/a/17289991/4168169
read.tcsv = function(file, header=TRUE, sep=",", ...) {
  
  n = max(count.fields(file, sep=sep), na.rm=TRUE)
  x = readLines(file)
  
  .splitvar = function(x, sep, n) {
    var = unlist(strsplit(x, split=sep))
    length(var) = n
    return(var)
  }
  
  x = do.call(cbind, lapply(x, .splitvar, sep=sep, n=n))
  x = apply(x, 1, paste, collapse=sep) 
  ## empty strings are converted to NA
  out = read.csv(text=x, sep=sep, header=header, na.strings = "", ...)
  return(out)
  
}


# Additional file 2 of Zhang (2018, doi:10.1186/s12918-018-0530-9)
mods <- read.tcsv(
    file="https://static-content.springer.com/esm/art%3A10.1186%2Fs12918-018-0530-9/MediaObjects/12918_2018_530_MOESM2_ESM.csv",
    sep=";"
)

mods_selected <- gsub(" ", "", mods$modulenumberinfilename[
    mods$Modulenumberinpaper %in% c(6, 14, 36, 39)
])

n_mods <- length(mods_selected)
genes_selected <- list()
all_genes <- character(0)

# Additional file 1 of Zhang (2018, doi:10.1186/s12918-018-0530-9)
httr::GET(
    "https://static-content.springer.com/esm/art%3A10.1186%2Fs12918-018-0530-9/MediaObjects/12918_2018_530_MOESM1_ESM.xlsx",
    httr::write_disk(temp_file <- tempfile(fileext = ".xlsx"))
)


for (i in 1:n_mods) {
  
  genes_selected[[i]] <- readxl::read_xlsx(
    path = temp_file, sheet = mods_selected[i]
  )[[2]]
  
  all_genes <- c(all_genes, genes_selected[[i]])
  
}


unlink(temp_file)

print("Names of the selected genes:")
print(genes_selected)

tmp <- RTCGA::expressionsTCGA(
  RTCGA.mRNA::BRCA.mRNA, RTCGA.mRNA::OV.mRNA, extract.cols = all_genes
)

tmp$dataset <- gsub(pattern = ".mRNA", replacement = "",  tmp$dataset)
tmp$dataset <- gsub(pattern = "RTCGA.mRNA::", replacement = "",  tmp$dataset)
tmp$dataset <- gsub(pattern = "RTCGA::", replacement = "",  tmp$dataset)

# Save the data as a CSV file for use with `gene.py`.
write.table(
  x = tmp[complete.cases(tmp), c("dataset", all_genes)],
  file = "gene.csv",
  sep = ",",
  row.names = FALSE,
  col.names = TRUE
)

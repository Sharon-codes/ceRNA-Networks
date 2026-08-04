import os
import re
import tarfile
import gzip
import zipfile
import io
import glob
import urllib.request
import logging
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TCRDataPipeline:
    def __init__(self, dataset_dir="./Dataset", output_dir="./Processed", dextramer_threshold=10):
        """
        Initializes the TCR data preprocessing pipeline.
        
        Args:
            dataset_dir (str): Path to the folder containing raw dataset files.
            output_dir (str): Path to the folder where output files will be saved.
            dextramer_threshold (int): UMI count cutoff for calling positive dextramer bindings in T-PLL.
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.dextramer_threshold = dextramer_threshold
        self.shapes_tracker = {}
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_vdjdb_if_missing(self):
        """
        Downloads the VDJdb database release from GitHub if files do not exist locally.
        """
        txt_path = os.path.join(self.dataset_dir, "vdjdb.txt")
        meta_path = os.path.join(self.dataset_dir, "vdjdb.meta.txt")
        
        if os.path.exists(txt_path) and os.path.exists(meta_path):
            logger.info("VDJdb local files found.")
            return
            
        logger.info("VDJdb files missing locally. Programmatically downloading release ZIP from GitHub...")
        url = "https://github.com/antigenomics/vdjdb-db/releases/download/2023-06-01/vdjdb-2023-06-01.zip"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                zip_data = response.read()
                
            logger.info("Extracting VDJdb release ZIP...")
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                for name in z.namelist():
                    if name.endswith("vdjdb.txt"):
                        with open(txt_path, "wb") as f:
                            f.write(z.read(name))
                    elif name.endswith("vdjdb.meta.txt"):
                        with open(meta_path, "wb") as f:
                            f.write(z.read(name))
            logger.info("VDJdb successfully downloaded and extracted.")
        except Exception as e:
            logger.error(f"Error occurred while downloading VDJdb: {e}")
            raise e
            
    def standardize_hla(self, hla):
        """
        Standardizes an HLA allele string into standard 4-digit G-group format (e.g. A*02:01).
        
        Args:
            hla (str): Raw HLA allele string.
            
        Returns:
            str: Standardized HLA string or original if unparseable.
        """
        if not isinstance(hla, str) or pd.isna(hla):
            return None
        hla = hla.strip()
        
        # Split by separator (like pipe or comma) and use the first allele
        hla = re.split(r"[|,\s]+", hla)[0]
        
        # Remove leading HLA- or HLA_ prefix
        hla = re.sub(r"^(HLA[-_])", "", hla, flags=re.IGNORECASE)
        
        # Normalize Cw to C
        hla = re.sub(r"^Cw", "C", hla, flags=re.IGNORECASE)
        
        # Pattern 1: Match standard formats like A*02:01 or A02:01 or A*2:01 or A2:01
        match = re.match(r"^([A-Z\d]+)\*?(\d{1,2}):(\d{2})", hla, re.IGNORECASE)
        if match:
            letter, allele_group, protein = match.groups()
            allele_group = f"{int(allele_group):02d}"
            letter_upper = letter.upper()
            if letter_upper == "DR":
                letter_upper = "DRB1"
            elif letter_upper == "DQ":
                letter_upper = "DQB1"
            return f"{letter_upper}*{allele_group}:{protein}"
            
        # Pattern 3: Match short formats (e.g. A2, B27, A02, DR1, DR15, DQ2)
        # Using letters-only for letter group to avoid greedily matching DR15 as DR1 + 5
        match = re.match(r"^([A-Z]+)\*?(\d{1,2})$", hla, re.IGNORECASE)
        if match:
            letter, digit = match.groups()
            allele_group = f"{int(digit):02d}"
            letter_upper = letter.upper()
            if letter_upper == "DR":
                letter_upper = "DRB1"
            elif letter_upper == "DQ":
                letter_upper = "DQB1"
            return f"{letter_upper}*{allele_group}:01"
            
        # Pattern 2: Match formats without colon (e.g. A0201 or A*0201 or A201)
        match = re.match(r"^([A-Z]+)\*?(\d{1,2})(\d{2})$", hla, re.IGNORECASE)
        if match:
            letter, allele_group, protein = match.groups()
            allele_group = f"{int(allele_group):02d}"
            letter_upper = letter.upper()
            if letter_upper == "DR":
                letter_upper = "DRB1"
            elif letter_upper == "DQ":
                letter_upper = "DQB1"
            return f"{letter_upper}*{allele_group}:{protein}"
            
        return hla

    def canonical_aa_qc(self, seq):
        """
        Checks if a sequence consists entirely of canonical amino acids.
        """
        if not isinstance(seq, str) or pd.isna(seq):
            return False
        seq = seq.strip().upper()
        if not seq:
            return False
        canonical_chars = set("ACDEFGHIKLMNPQRSTVWY")
        return all(char in canonical_chars for char in seq)
        
    def validate_cdr3_anchors(self, seq):
        """
        Enforces canonical CDR3 anchoring: starts with 'C' (Cysteine) 
        and ends with 'F' (Phenylalanine) or 'W' (Tryptophan).
        """
        if not isinstance(seq, str) or pd.isna(seq):
            return False
        seq = seq.strip().upper()
        if len(seq) < 2:
            return False
        return seq[0] == "C" and seq[-1] in ("F", "W")

    def process_vdjdb(self):
        """
        Reads, cross-references, filters, and pairs VDJdb records.
        """
        self.download_vdjdb_if_missing()
        txt_path = os.path.join(self.dataset_dir, "vdjdb.txt")
        meta_path = os.path.join(self.dataset_dir, "vdjdb.meta.txt")
        
        df = pd.read_csv(txt_path, sep="\t")
        logger.info(f"Loaded VDJdb. Initial shape: {df.shape}")
        
        # Verify schema against meta (just checking file presence, as it's standard)
        assert os.path.exists(meta_path), "VDJdb metadata file is missing!"
        
        # Filter strictly for vdjdb.score == 3
        df_filtered = df[df["vdjdb.score"] == 3]
        logger.info(f"Filtered VDJdb for score == 3. Shape: {df_filtered.shape}")
        assert len(df_filtered) > 0, "No records found in VDJdb with score == 3!"
        
        # Pair TRA and TRB records sharing the same complex.id (excluding complex.id == 0)
        paired = df_filtered[df_filtered["complex.id"] > 0]
        unpaired = df_filtered[df_filtered["complex.id"] == 0]
        
        records = []
        
        # Group and pair
        for complex_id, group in paired.groupby("complex.id"):
            tra = group[group["gene"] == "TRA"]
            trb = group[group["gene"] == "TRB"]
            
            if not trb.empty:
                cdr3_beta = trb.iloc[0]["cdr3"]
                cdr3_alpha = tra.iloc[0]["cdr3"] if not tra.empty else None
                peptide = trb.iloc[0]["antigen.epitope"]
                hla = trb.iloc[0]["mhc.a"]
                
                records.append({
                    "cdr3_beta": cdr3_beta,
                    "cdr3_alpha": cdr3_alpha,
                    "peptide": peptide,
                    "hla_allele": hla,
                    "disease": str(trb.iloc[0]["antigen.species"]).strip() if not pd.isna(trb.iloc[0]["antigen.species"]) else "Unknown"
                })
                
        # Unpaired (only keep TRB records, set alpha to None)
        for _, row in unpaired[unpaired["gene"] == "TRB"].iterrows():
            records.append({
                "cdr3_beta": row["cdr3"],
                "cdr3_alpha": None,
                "peptide": row["antigen.epitope"],
                "hla_allele": row["mhc.a"],
                "disease": str(row["antigen.species"]).strip() if not pd.isna(row["antigen.species"]) else "Unknown"
            })
            
        df_out = pd.DataFrame(records)
        logger.info(f"Processed VDJdb. Structured shape: {df_out.shape}")
        self.shapes_tracker["VDJdb_Structured"] = df_out.shape
        return df_out

    def process_mcpas_tcr(self):
        """
        Reads, filters, and standardizes McPAS-TCR records.
        """
        file_path = os.path.join(self.dataset_dir, "McPAS-TCR.csv")
        df = pd.read_csv(file_path, encoding="latin1")
        logger.info(f"Loaded McPAS-TCR. Initial shape: {df.shape}")
        
        # Filter for Human species
        df_filtered = df[df["Species"] == "Human"]
        logger.info(f"Filtered McPAS-TCR for Species == 'Human'. Shape: {df_filtered.shape}")
        
        # Determine column names (handle standard and fallback naming)
        beta_col = "CDR3.beta.aa" if "CDR3.beta.aa" in df_filtered.columns else "CDR3beta"
        alpha_col = "CDR3.alpha.aa" if "CDR3.alpha.aa" in df_filtered.columns else "CDR3alpha"
        pep_col = "Epitope.peptide" if "Epitope.peptide" in df_filtered.columns else "Epitope peptide"
        mhc_col = "MHC"
        
        records = []
        for _, row in df_filtered.iterrows():
            cdr3_beta = row[beta_col]
            cdr3_alpha = row[alpha_col]
            peptide = row[pep_col]
            hla = row[mhc_col]
            
            # Convert NaN to None
            cdr3_beta = None if pd.isna(cdr3_beta) else str(cdr3_beta).strip()
            cdr3_alpha = None if pd.isna(cdr3_alpha) else str(cdr3_alpha).strip()
            peptide = None if pd.isna(peptide) else str(peptide).strip()
            hla = None if pd.isna(hla) else str(hla).strip()
            
            # Since beta is core, skip if beta is missing
            if cdr3_beta:
                records.append({
                    "cdr3_beta": cdr3_beta,
                    "cdr3_alpha": cdr3_alpha,
                    "peptide": peptide,
                    "hla_allele": hla,
                    "disease": str(row["Pathology"]).strip() if not pd.isna(row["Pathology"]) else "Unknown"
                })
                
        df_out = pd.DataFrame(records)
        logger.info(f"Processed McPAS-TCR. Structured shape: {df_out.shape}")
        self.shapes_tracker["McPAS-TCR_Structured"] = df_out.shape
        return df_out

    def process_cedar(self):
        """
        Reads, filters, and standardizes CEDAR receptors export.
        """
        files = glob.glob(os.path.join(self.dataset_dir, "receptor_table_export_*.csv"))
        if not files:
            files = glob.glob(os.path.join(self.dataset_dir, "cedar_receptors.tsv"))
        if not files:
            raise FileNotFoundError("CEDAR dataset file not found in ./Dataset/!")
            
        file_path = files[0]
        logger.info(f"Loading CEDAR dataset from: {file_path}")
        
        # Read the file headers to determine format
        header_check = pd.read_csv(file_path, nrows=2, header=None)
        is_portal_export = "CEDAR Receptor ID" in header_check.values or "Organism IRI" in header_check.values
        
        if is_portal_export:
            # Portal MultiIndex export
            df = pd.read_csv(file_path, header=[0, 1], low_memory=False)
            logger.info(f"Loaded CEDAR portal export. Shape: {df.shape}")
            
            # Host Organism == 'Homo sapiens (human)'
            is_human_1 = df[("Chain 1", "Organism IRI")].astype(str).str.contains("9606", na=False)
            is_human_2 = df[("Chain 2", "Organism IRI")].astype(str).str.contains("9606", na=False)
            df_filtered = df[is_human_1 & is_human_2]
            logger.info(f"Filtered for Human host. Shape: {df_filtered.shape}")
            
            # MHC Class == 'Class I' (HLA-A, HLA-B, HLA-C alleles)
            mhc_series = df_filtered[("Assay", "MHC Allele Names")].astype(str)
            is_class_i = mhc_series.str.contains("HLA-A|HLA-B|HLA-C", case=False, na=False)
            df_filtered = df_filtered[is_class_i]
            logger.info(f"Filtered for MHC Class I. Shape: {df_filtered.shape}")
            
            # Extract standard columns
            records = []
            for _, row in df_filtered.iterrows():
                cdr3_beta = row[("Chain 2", "CDR3 Curated")]
                cdr3_alpha = row[("Chain 1", "CDR3 Curated")]
                peptide = row[("Epitope", "Name")]
                hla = row[("Assay", "MHC Allele Names")]
                
                cdr3_beta = None if pd.isna(cdr3_beta) else str(cdr3_beta).strip()
                cdr3_alpha = None if pd.isna(cdr3_alpha) else str(cdr3_alpha).strip()
                peptide = None if pd.isna(peptide) else str(peptide).strip()
                hla = None if pd.isna(hla) else str(hla).strip()
                
                if cdr3_beta:
                    organism = str(row[("Epitope", "Source Organism")]).strip() if not pd.isna(row[("Epitope", "Source Organism")]) else ""
                    molecule = str(row[("Epitope", "Source Molecule")]).strip().lower() if not pd.isna(row[("Epitope", "Source Molecule")]) else ""
                    if "homo sapiens" in organism.lower():
                        disease = "Melanoma" if ("melanoma" in molecule or "pmel" in molecule) else "Cancer"
                    else:
                        disease = organism if organism else "Unknown"
                    records.append({
                        "cdr3_beta": cdr3_beta,
                        "cdr3_alpha": cdr3_alpha,
                        "peptide": peptide,
                        "hla_allele": hla,
                        "disease": disease
                    })
            df_out = pd.DataFrame(records)
        else:
            # Standard flat CEDAR format
            df = pd.read_csv(file_path, sep=None, engine="python")
            logger.info(f"Loaded CEDAR standard flat file. Shape: {df.shape}")
            
            df_filtered = df[
                (df["Host Organism"] == "Homo sapiens (human)") &
                (df["MHC Class"] == "Class I")
            ]
            logger.info(f"Filtered for Human host & Class I. Shape: {df_filtered.shape}")
            
            org_col = "Epitope Organism" if "Epitope Organism" in df_filtered.columns else "Host Organism"
            df_out = pd.DataFrame({
                "cdr3_beta": df_filtered["Chain2 CDR3 Curated"],
                "cdr3_alpha": df_filtered["Chain1 CDR3 Curated"],
                "peptide": df_filtered["Epitope Linear Sequence"],
                "hla_allele": df_filtered["MHC Restriction"],
                "disease": df_filtered[org_col].apply(lambda x: "Cancer" if "homo sapiens" in str(x).lower() else str(x) if pd.notna(x) else "Unknown")
            })
            
        logger.info(f"Processed CEDAR. Structured shape: {df_out.shape}")
        self.shapes_tracker["CEDAR_Structured"] = df_out.shape
        return df_out

    def process_donor1(self):
        """
        Merges contig annotations and binarized matrix for Donor 1, 
        isolating positive binders and pairing TCR chains.
        """
        contig_path = os.path.join(self.dataset_dir, "vdj_v1_hs_aggregated_donor1_all_contig_annotations.csv")
        matrix_path = os.path.join(self.dataset_dir, "vdj_v1_hs_aggregated_donor1_binarized_matrix.csv")
        
        logger.info(f"Loading Donor 1 files: {contig_path} and {matrix_path}")
        df_c = pd.read_csv(contig_path)
        df_m = pd.read_csv(matrix_path)
        
        # Pair TCRs per cell barcode in contig annotations
        df_c_cell = df_c[(df_c["is_cell"] == "True") | (df_c["is_cell"] == True)]
        df_c_cell = df_c_cell[(df_c_cell["productive"] == "True") | (df_c_cell["productive"] == True)]
        
        tcr_dict = {}
        for barcode, group in df_c_cell.groupby("barcode"):
            tra = group[group["chain"] == "TRA"]
            trb = group[group["chain"] == "TRB"]
            if not trb.empty:
                cdr3_beta = trb.iloc[0]["cdr3"]
                cdr3_alpha = tra.iloc[0]["cdr3"] if not tra.empty else None
                tcr_dict[barcode] = (cdr3_alpha, cdr3_beta)
                
        # Find binder columns ending with _binder (excluding control binder columns)
        binder_cols = [c for c in df_m.columns if c.endswith("_binder") and not "_NC_binder" in c]
        logger.info(f"Found {len(binder_cols)} dextramer binder columns in Donor 1.")
        
        # Melt matrix to keep only active bindings (value == 1)
        df_m_binders = df_m[["barcode"] + binder_cols]
        df_m_melt = df_m_binders.melt(id_vars="barcode", value_vars=binder_cols, var_name="binder_col", value_name="is_binder")
        df_m_melt = df_m_melt[df_m_melt["is_binder"] == 1]
        
        records = []
        for _, row in df_m_melt.iterrows():
            barcode = row["barcode"]
            if barcode in tcr_dict:
                cdr3_alpha, cdr3_beta = tcr_dict[barcode]
                
                # Parse column name: e.g. A0201_SLLMWITQV_NY-ESO-1_Cancer_binder
                parts = row["binder_col"].replace("_binder", "").split("_")
                hla = parts[0]
                peptide = parts[1]
                
                records.append({
                    "cdr3_beta": cdr3_beta,
                    "cdr3_alpha": cdr3_alpha,
                    "peptide": peptide,
                    "hla_allele": hla,
                    "disease": "Healthy"
                })
                
        df_out = pd.DataFrame(records)
        logger.info(f"Processed Donor 1. Structured shape: {df_out.shape}")
        self.shapes_tracker["Donor1_Structured"] = df_out.shape
        return df_out

    def process_t_pll(self):
        """
        Loads the T-PLL 10x Genomics dataset, programmatically extracts the 
        matrix, streams UMI counts to identify positive binders, and pairs with TCR annotations.
        """
        contig_path = os.path.join(self.dataset_dir, "T_PLL_sorted_5pv2_nextgem_vdj_t_filtered_contig_annotations.csv")
        tar_path = os.path.join(self.dataset_dir, "T_PLL_sorted_5pv2_nextgem_count_filtered_feature_bc_matrix.tar.gz")
        
        logger.info(f"Loading T-PLL files: {contig_path} and {tar_path}")
        df_c = pd.read_csv(contig_path)
        
        # Pair TCR contigs by barcode
        df_c_cell = df_c[(df_c["is_cell"] == "True") | (df_c["is_cell"] == True)]
        df_c_cell = df_c_cell[(df_c_cell["productive"] == "True") | (df_c_cell["productive"] == True)]
        
        tcr_dict = {}
        for barcode, group in df_c_cell.groupby("barcode"):
            tra = group[group["chain"] == "TRA"]
            trb = group[group["chain"] == "TRB"]
            if not trb.empty:
                cdr3_beta = trb.iloc[0]["cdr3"]
                cdr3_alpha = tra.iloc[0]["cdr3"] if not tra.empty else None
                tcr_dict[barcode] = (cdr3_alpha, cdr3_beta)
                
        # Extract features and barcodes from sparse matrix tarball
        logger.info("Extracting features and barcodes from T-PLL archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            f_features = tar.extractfile("filtered_feature_bc_matrix/features.tsv.gz")
            with gzip.open(f_features, "rt") as f:
                features = pd.read_csv(f, sep="\t", header=None)
                
            f_barcodes = tar.extractfile("filtered_feature_bc_matrix/barcodes.tsv.gz")
            with gzip.open(f_barcodes, "rt") as f:
                barcodes = pd.read_csv(f, sep="\t", header=None)[0].tolist()
                
        # Dextramers feature indexes (Antibody Capture, excluding CTRL)
        dex_indices_set = set([
            i for i, r in features.iterrows() 
            if r[2] == "Antibody Capture" 
            and ("_A0" in r[0] or "_B0" in r[0] or "_A1" in r[0]) 
            and "CTRL" not in r[0]
        ])
        logger.info(f"Identified {len(dex_indices_set)} target dextramer panel features.")
        
        dex_meta = {}
        for idx in dex_indices_set:
            f_name = features.iloc[idx, 0]
            parts = f_name.split("_")
            hla = parts[-1]
            peptide = parts[-2]
            dex_meta[idx] = (hla, peptide)
            
        logger.info("Streaming matrix counts from Matrix Market archive (T-PLL)...")
        records = []
        with tarfile.open(tar_path, "r:gz") as tar:
            f_matrix = tar.extractfile("filtered_feature_bc_matrix/matrix.mtx.gz")
            with gzip.open(f_matrix, "rt") as f:
                # Skip Matrix Market header and comments
                for line in f:
                    if line.startswith("%"):
                        continue
                    break # This is the dimensions line
                    
                # Process count entries line-by-line
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    r = int(parts[0]) - 1
                    c = int(parts[1]) - 1
                    val = float(parts[2])
                    
                    if r in dex_indices_set and val >= self.dextramer_threshold:
                        barcode = barcodes[c]
                        if barcode in tcr_dict:
                            cdr3_alpha, cdr3_beta = tcr_dict[barcode]
                            hla, peptide = dex_meta[r]
                            records.append({
                                "cdr3_beta": cdr3_beta,
                                "cdr3_alpha": cdr3_alpha,
                                "peptide": peptide,
                                "hla_allele": hla,
                                "disease": "T-PLL"
                            })
                            
        df_out = pd.DataFrame(records)
        logger.info(f"Processed T-PLL. Structured shape: {df_out.shape}")
        self.shapes_tracker["T-PLL_Structured"] = df_out.shape
        return df_out

    def levenshtein_distance(self, s1, s2):
        """
        Computes the Levenshtein distance between two strings.
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def cluster_peptides(self, peptides, distance_threshold=2):
        """
        Clusters peptides based on sequence homology (Levenshtein distance <= threshold).
        """
        logger.info(f"Clustering {len(peptides)} unique peptides. Distance threshold: {distance_threshold}...")
        adj = defaultdict(list)
        
        # Build pairwise graph edges
        for i in tqdm(range(len(peptides)), desc="Calculating similarity distances"):
            for j in range(i + 1, len(peptides)):
                p1 = peptides[i]
                p2 = peptides[j]
                if self.levenshtein_distance(p1, p2) <= distance_threshold:
                    adj[p1].append(p2)
                    adj[p2].append(p1)
                    
        # Find connected components (homology clusters)
        visited = set()
        clusters = []
        for p in peptides:
            if p not in visited:
                component = []
                queue = [p]
                visited.add(p)
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(component)
        logger.info(f"Grouped into {len(clusters)} homologous peptide clusters.")
        return clusters

    def split_data(self, df):
        """
        De-duplicates the dataset and partitions it into train/val/test splits 
        ensuring entire peptide clusters stay within the same split to prevent leakage.
        """
        # Drop exact duplicates across triplets
        df_dedup = df.drop_duplicates(subset=["cdr3_beta", "peptide", "hla_allele"]).copy()
        logger.info(f"Dropped exact duplicates. Pre-de-duplication shape: {df.shape}, Post-de-duplication shape: {df_dedup.shape}")
        self.shapes_tracker["Post_DeDuplication"] = df_dedup.shape
        
        unique_peps = df_dedup["peptide"].unique().tolist()
        
        # Cluster peptides
        clusters = self.cluster_peptides(unique_peps, distance_threshold=2)
        
        # Partition clusters into Train (80%), Val (10%), Test (10%) dynamically
        np.random.seed(42)
        np.random.shuffle(clusters)
        
        train_peptides = []
        val_peptides = []
        test_peptides = []
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        for cluster in tqdm(clusters, desc="Allocating peptide clusters to splits"):
            cluster_count = df_dedup[df_dedup["peptide"].isin(cluster)].shape[0]
            assigned_so_far = train_count + val_count + test_count
            
            if assigned_so_far == 0:
                train_peptides.extend(cluster)
                train_count += cluster_count
            else:
                # Assign to the split furthest behind its target ratio
                train_diff = 0.8 - (train_count / assigned_so_far)
                val_diff = 0.1 - (val_count / assigned_so_far)
                test_diff = 0.1 - (test_count / assigned_so_far)
                
                max_diff = max(train_diff, val_diff, test_diff)
                if max_diff == train_diff:
                    train_peptides.extend(cluster)
                    train_count += cluster_count
                elif max_diff == val_diff:
                    val_peptides.extend(cluster)
                    val_count += cluster_count
                else:
                    test_peptides.extend(cluster)
                    test_count += cluster_count
                    
        train_df = df_dedup[df_dedup["peptide"].isin(train_peptides)].copy()
        val_df = df_dedup[df_dedup["peptide"].isin(val_peptides)].copy()
        test_df = df_dedup[df_dedup["peptide"].isin(test_peptides)].copy()
        
        logger.info(f"Split results:")
        logger.info(f"  Train: {train_df.shape[0]} records ({train_df.shape[0]/df_dedup.shape[0]*100:.2f}%)")
        logger.info(f"  Val:   {val_df.shape[0]} records ({val_df.shape[0]/df_dedup.shape[0]*100:.2f}%)")
        logger.info(f"  Test:  {test_df.shape[0]} records ({test_df.shape[0]/df_dedup.shape[0]*100:.2f}%)")
        
        # Verify zero leakage (assertion checks)
        train_peps_set = set(train_df["peptide"])
        test_peps_set = set(test_df["peptide"])
        intersection = train_peps_set.intersection(test_peps_set)
        assert len(intersection) == 0, f"Peptide leakage detected! Intersection: {intersection}"
        
        for p_test in test_peps_set:
            for p_train in train_peps_set:
                dist = self.levenshtein_distance(p_test, p_train)
                assert dist > 2, f"Homology leakage detected! Test: '{p_test}' and Train: '{p_train}' (dist={dist})"
                
        logger.info("Zero-leakage checks complete: no peptide or homologous peptide overlaps between splits.")
        return train_df, val_df, test_df

    def run(self):
        """
        Executes the entire curation pipeline.
        """
        logger.info("Starting TCR Data Preprocessing Pipeline...")
        
        # 1. Parse all datasets
        df_vdj = self.process_vdjdb()
        df_mcp = self.process_mcpas_tcr()
        df_cedar = self.process_cedar()
        df_don1 = self.process_donor1()
        df_tpll = self.process_t_pll()
        
        # 2. Combine datasets
        logger.info("Combining datasets...")
        df_all = pd.concat([df_vdj, df_mcp, df_cedar, df_don1, df_tpll], ignore_index=True)
        logger.info(f"Combined dataset shape: {df_all.shape}")
        self.shapes_tracker["Combined"] = df_all.shape
        assert df_all.shape[0] > 0, "Combined dataset contains no rows!"
        
        # 3. Standardization & Harmonization
        logger.info("Standardizing HLA alleles...")
        df_all["hla_allele"] = df_all["hla_allele"].apply(self.standardize_hla)
        
        # Drop rows missing required fields (cdr3_beta, peptide, hla_allele)
        before_drop = df_all.shape[0]
        df_all = df_all.dropna(subset=["cdr3_beta", "peptide", "hla_allele"]).copy()
        df_all["disease"] = df_all["disease"].fillna("Unknown").astype(str).str.strip()
        logger.info(f"Dropped rows with missing required columns: {before_drop - df_all.shape[0]} rows dropped. Shape: {df_all.shape}")
        
        # 4. String Quality Control (QC)
        logger.info("Applying Quality Control (QC) filters...")
        # Check canonical amino acids
        mask_canonical = (
            df_all["cdr3_beta"].apply(self.canonical_aa_qc) & 
            df_all["peptide"].apply(self.canonical_aa_qc) & 
            df_all["cdr3_alpha"].apply(lambda x: self.canonical_aa_qc(x) if pd.notna(x) and x is not None else True)
        )
        df_all = df_all[mask_canonical].copy()
        logger.info(f"Applied canonical amino acid filter. Shape: {df_all.shape}")
        self.shapes_tracker["QC_Canonical_AA"] = df_all.shape
        
        # Enforce canonical CDR3 anchoring (starts with C, ends with F/W)
        mask_anchored = (
            df_all["cdr3_beta"].apply(self.validate_cdr3_anchors) & 
            df_all["cdr3_alpha"].apply(lambda x: self.validate_cdr3_anchors(x) if pd.notna(x) and x is not None else True)
        )
        df_all = df_all[mask_anchored].copy()
        logger.info(f"Applied canonical CDR3 anchoring validation. Shape: {df_all.shape}")
        self.shapes_tracker["QC_Anchored"] = df_all.shape
        assert df_all.shape[0] > 0, "No records survived QC filter steps!"
        
        # 5. De-duplication & Leakage-free split
        train_df, val_df, test_df = self.split_data(df_all)
        
        # 6. Save final data partitions
        train_path = os.path.join(self.output_dir, "train.csv")
        val_path = os.path.join(self.output_dir, "val.csv")
        test_path = os.path.join(self.output_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved dataset partitions: {train_path}, {val_path}, {test_path}")
        
        # 7. Save shapes tracking
        self.shapes_tracker["Final_Train"] = train_df.shape
        self.shapes_tracker["Final_Val"] = val_df.shape
        self.shapes_tracker["Final_Test"] = test_df.shape
        
        tracking_path = os.path.join(self.output_dir, "shapes_tracking.json")
        with open(tracking_path, "w") as f:
            json.dump(self.shapes_tracker, f, indent=4)
        logger.info(f"Saved shape tracking summary to: {tracking_path}")
        
        logger.info("TCR Preprocessing Pipeline execution finished successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modular Preprocessing Pipeline for TCR-Peptide prediction")
    parser.add_argument("--dataset_dir", type=str, default="./Dataset", help="Path to raw Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./Processed", help="Path to save processed results")
    parser.add_argument("--threshold", type=int, default=10, help="Dextramer UMI count threshold for T-PLL sparse matrix")
    
    args = parser.parse_args()
    
    pipeline = TCRDataPipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dextramer_threshold=args.threshold
    )
    pipeline.run()

import os
import pandas as pd
import glob
import re
import argparse
from tqdm import tqdm
import random
from collections import Counter

# Common stop words in medical reports to ignore when finding keywords
STOP_WORDS = {
    'the', 'is', 'are', 'and', 'of', 'in', 'on', 'at', 'to', 'with', 'no', 'definite', 
    'abnormality', 'identified', 'this', 'slice', 'likely', 'representing', 'appears', 
    'otherwise', 'normal', 'within', 'limits', 'unremarkable', 'clear', 'intact', 
    'well', 'visualized', 'due', 'small', 'large', 'amount', 'prominent', 'somewhat',
    'irregular', 'structure', 'structures', 'soft', 'tissue', 'density', 'approx', 
    'measuring', 'cm', 'mm', 'diameter', 'seen', 'visible', 'present', 'evidence',
    'containing', 'loops', 'bowel', 'wall', 'thickness', 'obvious', 'significant',
    'scattered', 'throughout', 'near', 'level', 'body', 'midline', 'quadrant',
    'region', 'space', 'abnormality', 'abnormalities', 'image', 'shows', 'view',
    'cross-sectional', 'lower', 'upper', 'right', 'left', 'anterior', 'posterior',
    'lateral', 'medial', 'proximal', 'distal', 'superior', 'inferior', 'bilateral',
    'heterogeneous', 'homogeneous', 'hypodense', 'hyperdense', 'isodense', 'fluid',
    'free', 'air', 'gas', 'contrast', 'enhancement', 'phase', 'arterial', 'venous',
    'delayed', 'portal', 'liver', 'spleen', 'kidney', 'kidneys', 'pancreas', 'gallbladder',
    'stomach', 'intestine', 'colon', 'rectum', 'bladder', 'prostate', 'uterus', 'ovary',
    'adrenal', 'gland', 'aorta', 'cava', 'vein', 'artery', 'vessle', 'skeleton', 'bone',
    'spine', 'vertebra', 'rib', 'pelvis', 'hip', 'femur', 'lung', 'heart', 'mediastinum',
    'pleural', 'peritoneum', 'mesentery', 'omentum', 'abdominal', 'pelvic', 'chest',
    'thorax', 'diaphragm', 'muscle', 'psoas', 'wall', 'fat', 'subcutaneous', 'skin',
    'lesion', 'mass', 'nodule', 'cyst', 'calculus', 'stone', 'calcification', 'hernia',
    'obstruction', 'dilatation', 'thickening', 'inflammation', 'infection', 'abscess',
    'fluid', 'ascites', 'hemorrhage', 'bleed', 'thrombosis', 'embolism', 'infarct',
    'ischemia', 'perforation', 'pneumoperitoneum', 'collection', 'infiltration',
    'metastasis', 'tumor', 'cancer', 'neoplasm', 'lymphadenopathy', 'node', 'nodes'
}

# Potential findings to look for (curated list based on common abdomen CT findings)
POTENTIAL_LABELS = [
    'Ascites', 'Fluid', 'Free Air', 'Pneumoperitoneum',
    'Lesion', 'Mass', 'Nodule', 'Cyst', 'Tumor', 'Metastasis',
    'Hernia', 'Obstruction', 'Dilatation', 'Distension', 'Volvulus',
    'Calculus', 'Stone', 'Lithiasis', 'Calcification',
    'Thickening', 'Inflammation', 'Infection', 'Abscess', 'Fluid Collection',
    'Hemorrhage', 'Bleeding', 'Hematoma',
    'Thrombosis', 'Embolism', 'Infarct', 'Ischemia',
    'Perforation', 'Rupture',
    'Fracture', 'Dislocation',
    'Lymphadenopathy', 'Enlarged Node',
    'Anomaly', 'Deformity',
    'Fatty Liver', 'Steatosis', 'Cirrhosis',
    'Hydronephrosis', 'Hydroureter',
    'Cholecystitis', 'Cholelithiasis',
    'Appendicitis', 'Diverticulitis', 'Colitis',
    'Pancreatitis',
    'Splenomegaly', 'Hepatomegaly'
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def extract_labels(text, conditions):
    labels = []
    text = clean_text(text)
    for condition in conditions:
        # Simple keyword matching
        sub_conditions = [c.strip().lower() for c in condition.split('/')]
        found = 0
        for sub_c in sub_conditions:
            if sub_c in text:
                found = 1
                break
        labels.append(found)
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports_dir', type=str, default='dataset/reports', help="Directory containing report CSVs")
    parser.add_argument('--images_dir', type=str, default='dataset/ct_dataset', help="Directory containing patient image folders")
    parser.add_argument('--output_csv', type=str, default='dataset/ct_processed_classes.csv', help="Output CSV path")
    args = parser.parse_args()

    print(f"Scanning reports in {args.reports_dir}...")
    report_files = glob.glob(os.path.join(args.reports_dir, '*.csv'))
    
    data = []
    
    for report_file in tqdm(report_files, desc="Processing Reports"):
        basename = os.path.basename(report_file)
        # Assuming filename format is patient_X.csv
        # patient_1.csv -> Patient 1
        patient_id_str = os.path.splitext(basename)[0].replace('patient_', 'Patient ')
        
        # Try to match folder name format (Patient X)
        # Note: glob might return patient_10.csv before patient_2.csv if not sorted naturally
        
        # Robust folder finding
        # Search for folder that contains 'Patient' and the number ID
        # Extract number
        try:
           patient_num = int(re.search(r'\d+', basename).group())
           patient_folder_pattern = f"Patient {patient_num}"
        except:
           print(f"Skipping {basename} - cannot parse patient number")
           continue
           
        patient_img_dir = os.path.join(args.images_dir, f"Patient {patient_num}")
        
        # Case insensitive check
        if not os.path.exists(patient_img_dir):
            found = False
            for d in os.listdir(args.images_dir):
                if d.lower() == f"patient {patient_num}".lower() or d.lower() == f"patient_{patient_num}".lower():
                     patient_img_dir = os.path.join(args.images_dir, d)
                     found = True
                     break
            if not found:
                #print(f"Warning: Image directory not found for Patient {patient_num}")
                continue

        df = pd.read_csv(report_file)
        
        # Get images
        image_files = sorted(glob.glob(os.path.join(patient_img_dir, '*.jpg')))
        if not image_files:
            continue
            
        for _, row in df.iterrows():
            slice_id = row['slice_id']
            findings = row['findings']
            
            # Map slice to image
            # Assume slice_0001 -> 1 -> index 0
            try:
                slice_num = int(re.search(r'\d+', slice_id).group())
            except:
                continue
                
            if slice_num - 1 < len(image_files):
                image_path = image_files[slice_num - 1]
            else:
                 # Try matching
                 formatted_num = f"{slice_num:05d}"
                 match = [f for f in image_files if formatted_num in f]
                 if match:
                     image_path = match[0]
                 else:
                     continue

            # Generate labels
            labels = extract_labels(findings, POTENTIAL_LABELS)
            
            # If no specific labels found, but it says "Abnormal", maybe add a generic "Abnormal" label?
            # Or reliance on "No Finding" implicit?
            # Let's add "No Finding" explicitly
            
            is_normal = 1 if "no definite abnormality identified" in clean_text(findings) else 0
            
            entry = {
                'id': f"Patient_{patient_num}_{slice_id}",
                'Path': image_path,
                'findings': findings,
                'No Finding': is_normal
            }
            
            for col, val in zip(POTENTIAL_LABELS, labels):
                entry[col] = val
                
            data.append(entry)

    # Convert to DataFrame
    out_df = pd.DataFrame(data)
    
    # Split
    unique_ids = out_df['id'].unique()
    random.seed(42)
    shuffled_indices = list(range(len(out_df)))
    random.shuffle(shuffled_indices)
    
    n = len(out_df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    out_df.loc[shuffled_indices[:train_end], 'split'] = 'train'
    out_df.loc[shuffled_indices[train_end:val_end], 'split'] = 'val'
    out_df.loc[shuffled_indices[val_end:], 'split'] = 'test'
    
    print(f"Saving to {args.output_csv}...")
    out_df.to_csv(args.output_csv, index=False)
    print(f"Done. Saved {len(out_df)} records.")

if __name__ == '__main__':
    main()

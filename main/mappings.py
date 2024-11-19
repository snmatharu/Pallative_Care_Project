specialty_to_category = {
    # Oncology and Cancer Care
    "__Medical oncology": "Oncology and Cancer Care",
    "__Hematology": "Oncology and Cancer Care",
    "__Gynecological oncology": "Oncology and Cancer Care",
    "__General surgical oncology": "Oncology and Cancer Care",
    "__Hematology/oncology \x97 Pediatrics": "Oncology and Cancer Care",
    "_Radiation oncology": "Oncology and Cancer Care",
    "_Otolaryngology \x97 Head and neck surgery": "Oncology and Cancer Care",
    # Surgical Interventions
    "_Cardiac surgery": "Surgical Interventions",
    "_General surgery": "Surgical Interventions",
    "__Pediatric surgery": "Surgical Interventions",
    "_Neurosurgery": "Surgical Interventions",
    "_Obstetrics and gynecology": "Surgical Interventions",
    "_Ophthalmology": "Cataract Surgery",  # Special case, could also be "Surgical Interventions"
    "_Orthopedic surgery": "Surgical Interventions",
    "_Otolaryngology \x97 Head and neck surgery": "Surgical Interventions",
    "_Plastic surgery": "Surgical Interventions",
    "_Urology": "Surgical Interventions",
    "_Vascular surgery": "Surgical Interventions",
    "__Colorectal surgery": "Surgical Interventions",

    # Diagnostic Imaging
    "_Diagnostic radiology": "Diagnostic Imaging",
    "_Medical genetics and genomics": "Diagnostic Imaging",  # Generally not but included for completeness
    "_Neurology": "Diagnostic Imaging",  # In the context of Electroencephalography
    "_Nuclear medicine": "Diagnostic Imaging",
    "__Pediatric radiology": "Diagnostic Imaging",
    "_Diagnostic and molecular pathology": "Diagnostic Imaging",
    "_Diagnostic and clinical pathology": "Diagnostic Imaging",
    "_Hematological pathology": "Diagnostic Imaging",
    "__Forensic pathology": "Diagnostic Imaging",

    # Emergency and Critical Care
    "_Emergency medicine": "Emergency and Critical Care",
    "__Critical care medicine": "Emergency and Critical Care",
    "__Emergency family medicine": "Emergency and Critical Care",
    "__Emergency medicine \x97 Pediatrics": "Emergency and Critical Care",
    "__Critical care medicine \x97 Pediatrics": "Emergency and Critical Care",

    # Cataract Surgery
    "_Ophthalmology": "Cataract Surgery",

    # Defaulting others to Other if not falling into the above categories
    "All physicians": "Essentials",
    "All specialists": "Essentials",
    "__Family medicine": "Essentials",
    "__General practice": "Essentials",
    "_Anesthesiology": "Essentials",
    "_Dermatology": "Other",
    "_Internal medicine": "Essentials",
    "__Cardiology": "Other",
    "__Clinical immunology and allergy": "Essentials",
    "__Endocrinology and metabolism": "Essentials",
    "__Gastroenterology": "Essentials",
    "__Geriatric medicine": "Essentials",
    "__Infectious diseases": "Essentials",
    "__Nephrology": "Other",
    "__Occupational medicine": "Other",
    "__Respirology": "Essentials",
    "__Rheumatology": "Other",
    "_Public health and preventive medicine": "Essentials",
    "_Physical medicine and rehabilitation": "Essentials",
    "_Psychiatry": "Essentials",
    
    "_Medical biochemistry": "Other",
    "_Medical microbiology": "Other",
    "_Neuropathology": "Other",

    # Additional pediatric specialties
    "_Pediatrics": "Other",
    "__Cardiology \x97 Pediatrics": "Other",
    "__Neonatal–perinatal medicine": "Other",
    "__Endocrinology and metabolism \x97 Pediatrics": "Other",
    "__Gastroenterology \x97 Pediatrics": "Other",
    "__Infectious diseases \x97 Pediatrics": "Other",
    "__Nephrology \x97 Pediatrics": "Other",
    "__Respirology \x97 Pediatrics": "Other",
    "__Rheumatology \x97 Pediatrics": "Other",
    "__Adolescent medicine \x97 Pediatrics": "Other",
    "__Child and adolescent psychiatry \x97 Pediatrics": "Other",
    "__General internal medicine": "Other",
    "__Forensic psychiatry": "Other",
    "__Maternal–fetal medicine": "Other",
    "__Neuroradiology": "Other",
    "__Developmental \x97 Pediatrics": "Other",
    "__Geriatric psychiatry": "Other",
    
    # Newly added specialties
    "__Electroencephalography": "Other",
    "__Cardiology \x97 Pediatrics": "Other",
    "__Neonatal\x96perinatal medicine": "Other",
    
    "Medical scientists": "Essentials",
    "__Palliative medicine": "Essentials",
    "__Clinical immunology and allergy \x97 Pediatrics": "Other",
    "__Endocrinology and metabolism \x97 Pediatrics": "Other",
    "__Gastroenterology \x97 Pediatrics": "Other",
    "__Hematology/oncology \x97 Pediatrics": "Other",
    "__Infectious diseases \x97 Pediatrics": "Other",
    "__Nephrology \x97 Pediatrics": "Other",
    "__Respirology \x97 Pediatrics": "Other",
    "__Rheumatology \x97 Pediatrics": "Other",
    "__Emergency medicine \x97 Pediatrics": "Other",
    "__Critical care medicine \x97 Pediatrics": "Other",
    "__Adolescent medicine \x97 Pediatrics": "Other",
    "__Child and adolescent psychiatry \x97 Pediatrics": "Other",
    "__Maternal\x96fetal medicine": "Essentials",
    "__Developmental \x97 Pediatrics": "Other"
}

province_map = {
    'N.L.': 'Newfoundland and Labrador',
    'P.E.I.': 'Prince Edward Island',
    'N.S.': 'Nova Scotia',
    'N.B.': 'New Brunswick',
    'Que.': 'Quebec',
    'Ont.': 'Ontario',
    'Man.': 'Manitoba',
    'Sask.': 'Saskatchewan',
    'Alta.': 'Alberta',
    'B.C.': 'British Columbia',
    'Y.T.': 'Yukon',
    'N.W.T.': 'Northwest Territories',
    'Nun.': 'Nunavut'
}

procedure_to_category = {
    "CT Scan": "Diagnostic Imaging",
    "MRI Scan": "Diagnostic Imaging",
    "Cataract Surgery": "Diagnostic Imaging",
    "Bladder Cancer Surgery": "Surgical Interventions",
    "Breast Cancer Surgery": "Surgical Interventions",
    "CABG": "Surgical Interventions",
    "Colorectal Cancer Surgery": "Surgical Interventions",
    "Hip Fracture Repair": "Surgical Interventions",
    "Hip Replacement": "Surgical Interventions",
    "Knee Replacement": "Surgical Interventions",
    "Lung Cancer Surgery": "Surgical Interventions",
    "Prostate Cancer Surgery": "Surgical Interventions",
    "Bladder Cancer Surgery": "Oncology and Cancer Care",
    "Breast Cancer Surgery": "Oncology and Cancer Care",
    "Colorectal Cancer Surgery": "Oncology and Cancer Care",
    "Lung Cancer Surgery": "Oncology and Cancer Care",
    "Prostate Cancer Surgery": "Oncology and Cancer Care",
    "Radiation Therapy": "Oncology and Cancer Care",
    "Hip Fracture Repair/Emergency and Inpatient": "Emergency and Critical Care"
}

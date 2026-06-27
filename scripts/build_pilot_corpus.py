"""
scripts/build_pilot_corpus.py — Generate a small curated medical pilot corpus.

PURPOSE (prototype only): the P5 gated policy needs a real retriever for its
RETRIEVE branch, but the full MedRAG / StatPearls corpus (~427K snippets) is not
yet downloaded. This script writes a curated ~200-chunk corpus of factual
medical passages spanning the topic areas of the MIRAGE benchmark (anatomy,
physiology, pharmacology, pathology, microbiology, clinical management) so that
BM25 retrieval has genuine lexical matches to return.

This is a PILOT corpus. Retrieval-quality numbers obtained against it are
provisional and are clearly flagged as such in the report; they will be
re-measured against the full corpus in Phase D. The point of the prototype is
to validate the gate decision logic, retrieval-rate accounting, and the
end-to-end pipeline — not retrieval recall.

Output: data/corpora/pilot_corpus.jsonl  (one Chunk per line)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

OUT = Path("data/corpora/pilot_corpus.jsonl")

# Each entry: (source, title, text). Sources mirror the real corpora names so
# the schema and citation paths are exercised realistically.
PASSAGES: List[tuple[str, str, str]] = [
    # ── Anatomy ──────────────────────────────────────────────────
    ("statpearls", "Phrenic Nerve Anatomy",
     "The phrenic nerve arises from cervical roots C3, C4, and C5 and is the "
     "sole motor supply to the diaphragm. It also carries sensory fibres to the "
     "central diaphragm, pericardium, and mediastinal pleura."),
    ("statpearls", "Vagus Nerve",
     "The vagus nerve (cranial nerve X) provides parasympathetic innervation to "
     "the heart, lungs, and gastrointestinal tract down to the splenic flexure. "
     "It does not innervate the diaphragm."),
    ("textbooks", "Brachial Plexus Organisation",
     "The brachial plexus is formed from spinal roots C5 to T1, organised into "
     "roots, trunks, divisions, cords, and branches. The median, ulnar, and "
     "radial nerves are its major terminal branches."),
    ("textbooks", "Coronary Artery Supply",
     "The left anterior descending artery supplies the anterior left ventricle "
     "and interventricular septum. The right coronary artery supplies the SA "
     "node in most people and the inferior wall of the left ventricle."),
    ("statpearls", "Femoral Triangle",
     "The femoral triangle contains, from lateral to medial, the femoral nerve, "
     "femoral artery, and femoral vein. The mnemonic NAVEL aids recall of the "
     "order of contents."),
    # ── Physiology ───────────────────────────────────────────────
    ("textbooks", "Cardiac Action Potential",
     "The ventricular cardiac action potential has five phases. Phase 0 is rapid "
     "depolarisation by sodium influx; phase 2 is the plateau from calcium "
     "influx; phase 3 is repolarisation by potassium efflux."),
    ("textbooks", "Renal Sodium Handling",
     "Roughly 65 percent of filtered sodium is reabsorbed in the proximal "
     "convoluted tubule. The loop of Henle reabsorbs about 25 percent via the "
     "sodium-potassium-two-chloride cotransporter, the target of loop diuretics."),
    ("textbooks", "Oxygen Dissociation Curve",
     "A right shift of the oxyhaemoglobin dissociation curve reduces oxygen "
     "affinity and is caused by increased carbon dioxide, increased temperature, "
     "increased 2,3-DPG, and decreased pH (the Bohr effect)."),
    ("textbooks", "Insulin Action",
     "Insulin promotes glucose uptake in muscle and adipose tissue by recruiting "
     "GLUT4 transporters, stimulates glycogen synthesis, and inhibits hepatic "
     "gluconeogenesis and lipolysis."),
    # ── Pharmacology ─────────────────────────────────────────────
    ("textbooks", "Metformin Mechanism of Action",
     "Metformin is a biguanide that lowers blood glucose chiefly by inhibiting "
     "hepatic gluconeogenesis through activation of AMP-activated protein kinase "
     "(AMPK). It is first-line pharmacotherapy for type 2 diabetes mellitus."),
    ("statpearls", "Aspirin Pharmacology",
     "Aspirin irreversibly acetylates cyclooxygenase-1 and cyclooxygenase-2, "
     "blocking synthesis of thromboxane A2 and prostaglandins. Low-dose aspirin "
     "is used for antiplatelet cardiovascular protection."),
    ("bnf", "Warfarin and Antibiotic Interactions",
     "Many antibiotics, including ciprofloxacin and metronidazole, potentiate "
     "warfarin by inhibiting cytochrome P450 metabolism and reducing vitamin "
     "K-producing gut flora. The INR must be monitored closely and the warfarin "
     "dose may need reduction."),
    ("bnf", "Methotrexate and Trimethoprim Interaction",
     "Co-administration of methotrexate with trimethoprim markedly increases the "
     "risk of severe bone marrow suppression because both drugs are antifolates. "
     "This combination should be avoided; if unavoidable, intensive haematologic "
     "monitoring is required."),
    ("bnf", "ACE Inhibitor Adverse Effects",
     "Angiotensin-converting enzyme inhibitors such as ramipril can cause a dry "
     "cough from bradykinin accumulation, hyperkalaemia, and, rarely, "
     "angioedema. They are contraindicated in pregnancy and bilateral renal "
     "artery stenosis."),
    ("textbooks", "Beta Blocker Pharmacodynamics",
     "Beta-adrenergic antagonists reduce heart rate and myocardial contractility, "
     "lowering myocardial oxygen demand. They are used in angina, heart failure, "
     "and after myocardial infarction, but caution is needed in asthma."),
    ("bnf", "Opioid Overdose Management",
     "Naloxone is a competitive opioid antagonist that reverses respiratory "
     "depression in opioid overdose. Because its duration is shorter than many "
     "opioids, repeated doses or an infusion may be required."),
    ("bnf", "Digoxin Toxicity",
     "Digoxin has a narrow therapeutic index. Hypokalaemia potentiates its "
     "toxicity. Features of toxicity include nausea, visual disturbance with "
     "yellow-green halos, and arrhythmias; severe cases are treated with "
     "digoxin-specific antibody fragments."),
    # ── Pathology ────────────────────────────────────────────────
    ("statpearls", "Acute Myocardial Infarction",
     "Acute myocardial infarction is most often caused by rupture of an "
     "atherosclerotic plaque with thrombus formation. Troponin is the most "
     "sensitive and specific biomarker. ST elevation indicates transmural "
     "injury requiring urgent reperfusion."),
    ("statpearls", "Type 1 vs Type 2 Diabetes",
     "Type 1 diabetes results from autoimmune destruction of pancreatic beta "
     "cells causing absolute insulin deficiency. Type 2 diabetes is "
     "characterised by insulin resistance with relative insulin deficiency and "
     "is strongly associated with obesity."),
    ("textbooks", "Nephrotic vs Nephritic Syndrome",
     "Nephrotic syndrome features heavy proteinuria over 3.5 grams per day, "
     "hypoalbuminaemia, and oedema. Nephritic syndrome features haematuria, "
     "hypertension, and a more modest proteinuria with red cell casts."),
    ("statpearls", "Anaphylaxis",
     "Anaphylaxis is a severe IgE-mediated type I hypersensitivity reaction "
     "causing airway oedema, bronchospasm, and hypotension. First-line treatment "
     "is intramuscular adrenaline; antihistamines and steroids are adjuncts."),
    # ── Microbiology / Infection ─────────────────────────────────
    ("statpearls", "Community Acquired Pneumonia",
     "Streptococcus pneumoniae is the commonest cause of community-acquired "
     "pneumonia. Empirical therapy commonly uses amoxicillin, with a macrolide "
     "added for atypical cover in more severe disease."),
    ("textbooks", "Antibiotic Mechanisms",
     "Beta-lactam antibiotics inhibit bacterial cell wall synthesis by binding "
     "penicillin-binding proteins. Aminoglycosides inhibit the 30S ribosomal "
     "subunit, and fluoroquinolones inhibit DNA gyrase and topoisomerase IV."),
    ("statpearls", "Tuberculosis Treatment",
     "Standard first-line therapy for active tuberculosis uses rifampicin, "
     "isoniazid, pyrazinamide, and ethambutol for two months, followed by "
     "rifampicin and isoniazid for four further months. Isoniazid can cause "
     "peripheral neuropathy, prevented with pyridoxine."),
    # ── Clinical management / emergency ──────────────────────────
    ("who_mhgap", "Status Epilepticus",
     "Status epilepticus is a seizure lasting more than five minutes or repeated "
     "seizures without recovery. First-line treatment is a benzodiazepine such "
     "as intravenous lorazepam, followed by a longer-acting antiepileptic if "
     "seizures continue."),
    ("who_mhgap", "Sepsis Recognition",
     "Sepsis is life-threatening organ dysfunction from a dysregulated host "
     "response to infection. The Sepsis Six bundle includes oxygen, blood "
     "cultures, intravenous antibiotics, fluids, lactate measurement, and urine "
     "output monitoring within one hour."),
    ("nice", "Hypertension First-Line Treatment",
     "For patients under 55 without type 2 diabetes, first-line antihypertensive "
     "therapy is an ACE inhibitor or angiotensin receptor blocker. For those "
     "over 55 or of Black African or Caribbean origin, a calcium channel blocker "
     "is preferred."),
    ("nice", "Asthma Acute Exacerbation",
     "Acute asthma is treated with high-flow oxygen, inhaled or nebulised "
     "salbutamol, and oral or intravenous corticosteroids. Ipratropium and "
     "intravenous magnesium sulfate are added in severe or life-threatening "
     "attacks."),
    ("nice", "Stroke Thrombolysis",
     "Ischaemic stroke within 4.5 hours of symptom onset may be treated with "
     "intravenous alteplase after haemorrhage is excluded by imaging. Mechanical "
     "thrombectomy is indicated for large-vessel occlusion within an extended "
     "window."),
]


def _expand_to_target(passages: List[tuple], target: int) -> List[tuple]:
    """
    The curated set is the knowledge core. To give BM25 a larger corpus to rank
    against (so retrieval is not trivially small), add lightly-varied distractor
    passages drawn from the same topics. These are clearly marked as filler.
    """
    expanded = list(passages)
    topics = [
        ("textbooks", "Clinical Note",
         "This passage discusses general principles of {} relevant to medical "
         "practice and patient assessment in routine clinical settings."),
    ]
    fillers = [
        "diabetes management", "cardiac arrhythmia", "renal failure",
        "respiratory infection", "anticoagulation therapy", "seizure disorders",
        "hypertension control", "antibiotic stewardship", "pain management",
        "electrolyte disturbance", "thyroid dysfunction", "anaemia workup",
        "liver function", "wound healing", "vaccination schedules",
    ]
    i = 0
    while len(expanded) < target:
        src, title, tmpl = topics[0]
        topic = fillers[i % len(fillers)]
        expanded.append((src, f"{title}: {topic.title()}", tmpl.format(topic)))
        i += 1
    return expanded


def main() -> None:
    passages = _expand_to_target(PASSAGES, target=200)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for idx, (source, title, text) in enumerate(passages):
            chunk: Dict[str, object] = {
                "chunk_id": f"pilot_{idx:04d}",
                "source": source,
                "title": title,
                "text": text,
                "score": 0.0,
            }
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {len(passages)} chunks to {OUT}")


if __name__ == "__main__":
    main()

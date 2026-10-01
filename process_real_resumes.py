#!/usr/bin/env python3
"""Process real labelled resumes through B0->B1->B2 and create B3 training dataset.

The real/ directory contains:
- .txt files: extracted text from PDFs (via some prior process)
- .labels.csv files: line-level annotations with header_type column

We need to:
1. Run B0->B1->B2 on each .txt file to get SegmentationResult with CandidateBlocks
2. Map the line-level CSV annotations to B2 block-level labels
3. Create a TrainingDataset for B3
"""

from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from resume_parser.classification import (
    LabelledBlock,
    LabelledDocument,
    SectionLabel,
    TrainingDataset,
)
from resume_parser.conversion import convert_bytes
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import segment_document


# Map from CSV header_type to our SectionLabel
CSV_TO_SECTION = {
    "summary_objective": SectionLabel.SUMMARY,
    "skills": SectionLabel.SKILLS,
    "experience": SectionLabel.EXPERIENCE,
    "education": SectionLabel.EDUCATION,
    "projects": SectionLabel.PROJECTS,
    "certifications": SectionLabel.CERTIFICATIONS,
    "awards": SectionLabel.AWARDS,
    "publications": SectionLabel.PUBLICATIONS,
    "languages": SectionLabel.LANGUAGES,
    "volunteering": SectionLabel.VOLUNTEERING,
    "interests": SectionLabel.INTERESTS,
    "references": SectionLabel.REFERENCES,
    "contact": SectionLabel.CONTACT,
    "other": SectionLabel.OTHER,
    "": SectionLabel.OTHER,  # blank = other
}


def load_annotations(csv_path: Path) -> dict[int, SectionLabel]:
    """Load line-level annotations from CSV.
    
    Returns mapping from line_id -> SectionLabel.
    Non-header lines inherit the section from the nearest preceding header.
    """
    annotations = {}
    current_section = SectionLabel.OTHER
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_id = int(row["line_id"])
            header_type = row["header_type"].strip().lower()
            
            if header_type:
                # This is a header line - update current section
                current_section = CSV_TO_SECTION.get(header_type, SectionLabel.OTHER)
            
            annotations[line_id] = current_section
    
    return annotations


def map_annotations_to_blocks(
    seg_result: Any,
    annotations: dict[int, SectionLabel],
) -> list[LabelledBlock]:
    """Map line-level annotations to B2 candidate blocks.
    
    For each block, determine the majority section label from its lines.
    Also record semantic spans if a block contains multiple sections.
    """
    # Build line_id -> annotation mapping for B1 lines
    # The B1 lines have line_ids like L000000, L000001, etc.
    # The CSV annotations use 0-based line indices from the original text file.
    # We need to map between them.
    
    # First, get the original text lines (before B1 normalization)
    # The B1 lines preserve the original text in raw_text
    # We can match by content
    
    # Actually, the B1 line indices (ln.index) correspond to the original line order
    # The CSV line_ids are also 0-based indices into the original text lines
    # So ln.index should match the CSV line_id for non-blank lines
    
    # Let's verify by building a mapping from B1 line index to block
    line_to_block = {}
    for block in seg_result.blocks:
        for lid in block.line_ids:
            line_to_block[lid] = block.block_id
    
    # Get annotations for each B1 line index
    block_annotations = defaultdict(list)  # block_id -> list of (line_id, section)
    
    for ln in seg_result.blocks[0].__class__.__module__:  # hack to get access to doc lines
        pass
    
    # Better approach: we need access to the original NormalizedDocument lines
    # The seg_result doesn't have the full line list, only blocks
    # We'll need to pass the doc lines separately
    
    return []


def process_real_resume(txt_path: Path, csv_path: Path) -> LabelledDocument | None:
    """Process a single real resume through B0->B1->B2 and create LabelledDocument."""
    try:
        # Read the text file
        text = txt_path.read_text(encoding="utf-8")
        
        # Run B0->B1->B2
        conv = convert_bytes(text.encode("utf-8"), filename=txt_path.name, content_type="text/plain")
        conv = conv.model_copy(update={"extracted_markdown": text, "extracted_char_count": len(text)})
        doc = normalize_document(conv)
        seg = segment_document(doc)
        
        # Load annotations
        annotations = load_annotations(csv_path)
        
        # Map annotations to blocks
        # B1 lines have indices 0, 1, 2... which correspond to CSV line_ids
        # But some lines might be blank or boilerplate and excluded from blocks
        
        # Build mapping from B1 line index -> section label
        line_sections = {}
        for ln in doc.lines:
            if ln.index in annotations:
                line_sections[ln.line_id] = annotations[ln.index]
        
        # For each block, collect the sections of its lines
        labelled_blocks = []
        for block in seg.blocks:
            if not block.line_ids:
                continue
            
            # Get sections for lines in this block
            block_sections = []
            for lid in block.line_ids:
                if lid in line_sections:
                    block_sections.append(line_sections[lid])
            
            if not block_sections:
                # No annotations for this block's lines - skip or use OTHER
                continue
            
            # Determine majority section
            from collections import Counter
            section_counts = Counter(block_sections)
            majority_section = section_counts.most_common(1)[0][0]
            
            # Check if block is mixed (multiple sections)
            semantic_spans = None
            if len(section_counts) > 1:
                # Block contains multiple sections - record spans
                spans = []
                current_section = block_sections[0]
                span_start_idx = 0
                span_start_lid = block.line_ids[0]
                
                for i, (lid, section) in enumerate(zip(block.line_ids, block_sections)):
                    if section != current_section:
                        # End previous span
                        spans.append((
                            span_start_lid,
                            block.line_ids[i-1],
                            current_section,
                        ))
                        # Start new span
                        current_section = section
                        span_start_idx = i
                        span_start_lid = lid
                
                # Final span
                spans.append((
                    span_start_lid,
                    block.line_ids[-1],
                    current_section,
                ))
                semantic_spans = spans
            
            labelled_blocks.append(LabelledBlock(
                document_id=seg.document_id,
                block_id=block.block_id,
                line_ids=block.line_ids,
                start_line_index=block.start_line_index,
                end_line_index=block.end_line_index,
                text=block.text,
                gold_section=majority_section,
                semantic_spans=semantic_spans,
            ))
        
        return LabelledDocument(
            document_id=seg.document_id,
            filename=txt_path.name,
            blocks=labelled_blocks,
        )
    
    except Exception as e:
        warnings.warn(f"Failed to process {txt_path}: {e}")
        return None


def main():
    real_dir = Path("/home/zoya_harmain/Resume_block_logic/real")
    
    # Get unique .txt files (skip _1.txt duplicates)
    txt_files = [f for f in real_dir.glob("*.txt") if not f.name.endswith("_1.txt")]
    print(f"Found {len(txt_files)} unique resumes to process")
    
    documents = []
    for txt_path in sorted(txt_files):
        csv_path = real_dir / f"{txt_path.stem}.labels.csv"
        if not csv_path.exists():
            print(f"  SKIP {txt_path.name}: no label file")
            continue
        
        print(f"  Processing {txt_path.name}...")
        doc = process_real_resume(txt_path, csv_path)
        if doc:
            documents.append(doc)
            print(f"    -> {len(doc.blocks)} labelled blocks")
        else:
            print(f"    -> FAILED")
    
    if not documents:
        print("No documents processed!")
        return
    
    # Create training dataset
    dataset = TrainingDataset(documents)
    print(f"\nCreated dataset: {dataset}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Save dataset info
    print("\nDocument details:")
    for doc in documents:
        sections = [b.gold_section.value for b in doc.blocks]
        print(f"  {doc.document_id} ({doc.filename}): {len(doc.blocks)} blocks - {sections}")
    
    return dataset


if __name__ == "__main__":
    main()
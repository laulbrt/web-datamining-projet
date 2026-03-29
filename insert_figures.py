"""
Insert the 3 figures into reports/final_report.pdf
- Fig 1 (pipeline)   -> inserted as new page after page 1
- Fig 3 (KB subgraph)-> inserted as new page after page 4
- Fig 2 (t-SNE)      -> inserted as new page after page 6
Output: reports/final_report_with_figures.pdf
"""
import fitz  # PyMuPDF
import os

SRC  = "reports/final_report.pdf"
OUT  = "reports/final_report_with_figures.pdf"
PAGE_W = 612
PAGE_H = 792

FIGURES = [
    # (image_path, insert_after_page_0indexed, caption)
    ("figures/fig1_pipeline.png",    0,
     "Figure 1 — Full pipeline from web crawling to KGE training."),
    ("figures/fig3_kb_subgraph.png", 4,   # will shift after fig1 insert
     "Figure 2 — Knowledge Base subgraph: key entities and relations (top 30 nodes)."),
    ("figures/fig2_tsne_embeddings.png", 7,  # will shift after fig1+fig3 inserts
     "Figure 3 — t-SNE projection of TransE entity embeddings (500 entities)."),
]

def make_figure_page(doc, img_path, caption):
    """Create a new PDF page with an image + caption centered."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    # background
    page.draw_rect(fitz.Rect(0, 0, PAGE_W, PAGE_H),
                   color=None, fill=(0.05, 0.07, 0.09))

    # load image
    img_rect = fitz.Rect(36, 50, PAGE_W - 36, PAGE_H - 80)
    page.insert_image(img_rect, filename=img_path, keep_proportion=True)

    # caption
    cap_rect = fitz.Rect(36, PAGE_H - 70, PAGE_W - 36, PAGE_H - 10)
    page.insert_textbox(
        cap_rect, caption,
        fontsize=9, fontname="helv",
        color=(0.7, 0.75, 0.8),
        align=fitz.TEXT_ALIGN_CENTER
    )
    return page

doc = fitz.open(SRC)
print(f"Opened: {SRC} ({len(doc)} pages)")

# Insert in reverse order so page indices stay valid
for img_path, after_page, caption in reversed(FIGURES):
    print(f"  Inserting {img_path} after page {after_page+1}...")
    # Create a temporary single-page doc
    tmp = fitz.open()
    make_figure_page(tmp, img_path, caption)
    # Insert into main doc
    doc.insert_pdf(tmp, start_at=after_page + 1)
    tmp.close()

doc.save(OUT, garbage=4, deflate=True)
doc.close()
print(f"\nSaved: {OUT} ({len(fitz.open(OUT))} pages)")

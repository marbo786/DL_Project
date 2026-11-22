# Dataset feasibility audit

The candidate review below uses public dataset cards. The local AIForge-Doc receipt audit follows the candidate review.

| Dataset | Published scope | Access and proposed engineering use |
| --- | --- | --- |
| AIForge-Doc v1 | 4,061 forged images, 812 published authentic test mirrors, masks and metadata. Gemini 2.5 Flash Image and Ideogram v2 Edit. | Hugging Face form, CC BY-NC-SA 4.0. Check whether each genuine training reference can be joined to its forged variants before constructing a balanced training set. |
| AIForge-Doc v2 | Card reports 3,066 GPT-Image-2 forgeries with authentic pairings and pixel masks. | Hugging Face form, CC BY-NC-SA 4.0. Join across versions by upstream collection and source identity. The v2 card describes reuse of v1 edit specifications. |
| DocTamper | Traditional document text tampering in LMDB, with localization annotations. | The repository lists Kaggle and Baidu and requires a password for non-commercial access. Treat as a distinct shift after reviewing scope and class availability. |
| DocDet ScamAI Crops | Derived positive and same-source negative crops, pair IDs, masks, mixed local and global edits and grouped splits. | Gated, CC BY-NC-SA 4.0. Crop-level evidence only. The derivative card and current upstream cards differ on authentic pairing availability. Verify against downloaded versions. |
| GPT4o-Receipt | Full-image synthesis. The paper benchmark contains 935 generated and 300 authentic receipts. The released directory describes a generated subset. | Gated, CC BY-NC-SA 4.0. Candidate out-of-scope probe. It does not provide local edit masks for the primary task. |

Before model work, record dataset revision and terms, audit source identity across all candidates, and check whether file format, encoding, dimensions, metadata and corpus identity predict the label. Isolate final shifted source groups from training and validation. Manual review is needed for uncertain and near-duplicate matches. Verify terms for authentic source corpora and any redistributed artifacts independently.

- [AIForge-Doc v1 card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v1)
- [AIForge-Doc v2 card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2)
- [DocTamper repository](https://github.com/qcf-568/DocTamper)
- [DocDet ScamAI Crops card](https://huggingface.co/datasets/tzj04/docdet-scamai-crops)
- [GPT4o-Receipt card](https://huggingface.co/datasets/Scam-AI/gpt4o-receipt)
TDoc-2.8M is an additional public research lead. Its card lists an MIT license, approximately 2.8 million tampered images and about 7.44 TB in compressed shards. Evaluate a small shard and verify source identities and genuine counterpart availability before treating it as a matched document benchmark. No TDoc files were used in the current release.

- [TDoc-2.8M dataset card](https://huggingface.co/datasets/MohamedDhouib1/TDoc-2.8M)
- [DocTamper paper](https://openaccess.thecvf.com/content/CVPR2023/html/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_CVPR_2023_paper.html)

## Local AIForge-Doc v2.0.1 receipt import

The full package contained 3,066 authentic, edited, and mask triplets. The importer found 2,648 receipt pairs and excluded 418 form pairs. The publisher's original train and test receipt source IDs were disjoint, but the manifest audit found byte-identical files under different IDs. Exact-image deduplication removed 31 source groups. It retained one complete source group per duplicate component and prioritized shifted test, in-domain test, validation, then training.

| Partition | CORD pairs | SROIE pairs | WildReceipt pairs | Total pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 616 | 175 | 0 | 791 |
| Validation | 168 | 60 | 0 | 228 |
| Test | 197 | 65 | 0 | 262 |
| Shifted test | 0 | 0 | 1,336 | 1,336 |

The in-domain test partition follows the publisher's test split. A stable source-ID hash assigns validation from the remaining CORD and SROIE receipts. WildReceipt is held out as a source-corpus shift. All edits use GPT-Image-2.

A seeded sample of 24 pairs per corpus had nonempty masks and different authentic and edited pixels. All 72 pairs matched exactly outside the provided masks. Before deduplication, every positive training and validation mask retained foreground at 112 × 160 under the model's nearest-neighbor resize. The median resized foreground area was 0.39% of image pixels for CORD and 1.45% for SROIE. Byte hashing does not catch a duplicate saved with a different encoding or a near-duplicate photograph.

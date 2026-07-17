# Project Dataset: Fashionpedia-1K Context Subset

## Exact scope

This project uses **1,000 images selected from the official Fashionpedia 2020 validation split**. The validation split contains 1,158 annotated images; 1,000 are retained after environment-first selection and preference for images with reliable garment colors and diverse garment types.

- Source: Fashionpedia 2020 validation annotations and official validation/test image archive
- Selected images: 1,000
- Garment and accessory instances: 4,337
- Split: 700 train, 151 validation, 149 test
- Seed: 42
- Metadata: `data/processed/fashionpedia_1000/metadata.jsonl`

The official Fashionpedia test-only images are excluded because they do not have garment annotations. The 46,781-image train+validation corpus is not used, keeping the project within the assignment's requested 500–1,000-image scale.

## The three primary axes

### 1. Environment

Fashionpedia does not provide office, urban-street, park, or home labels. Environments are derived with `openai/clip-vit-base-patch32` using four explicit scene prompts. Predictions are stored as `unknown` when confidence is below 0.40 or the top-two margin is below 0.08.

| Environment | Images | Share |
|---|---:|---:|
| Urban street | 569 | 56.9% |
| Office | 160 | 16.0% |
| Park | 91 | 9.1% |
| Home | 87 | 8.7% |
| Unknown | 93 | 9.3% |

This axis is not balanced. The source validation split has too few confidently classified park and home scenes to reach 250 examples per class. Results must therefore be reported per environment as well as overall.

### 2. Clothing type

Clothing types come directly from Fashionpedia instance annotations and segmentation masks. The subset contains 4,337 garment/accessory instances across 27 primary categories.

The largest categories are:

| Clothing type | Instances |
|---|---:|
| Shoe | 1,457 |
| Top / T-shirt / sweatshirt | 443 |
| Dress | 406 |
| Pants | 300 |
| Bag / wallet | 204 |
| Jacket | 167 |
| Belt | 154 |
| Skirt | 153 |

Shoes are heavily overrepresented. Tie is especially scarce with only three instances, so the assignment's red-tie query cannot be evaluated reliably without a targeted supplement.

### 3. Garment color

Fashionpedia does not contain native color labels. Color is derived separately for each garment by sampling pixels inside its official segmentation polygon and classifying the robust median RGB value into a fixed 13-color vocabulary:

`black, white, gray, red, orange, yellow, green, blue, navy, purple, pink, brown, beige`

Of 4,337 garment instances, 3,908 have a mask-derived color and 429 remain unknown.

| Color | Instances |
|---|---:|
| Black | 973 |
| Gray | 913 |
| Red | 445 |
| Brown | 347 |
| Orange | 272 |
| Navy | 253 |
| White | 210 |
| Blue | 147 |

Green (26) and yellow (40) are rare. Evaluation for yellow raincoats therefore needs deliberate query-level auditing or supplemental examples.

## Cross-axis analysis

The dashboard includes two important interaction plots:

- **Environment × clothing type:** detects combinations missing from particular settings.
- **Clothing type × garment color:** detects compositional gaps such as garments that occur mostly in black or gray.

These plots matter more than three independent histograms. A dataset may contain shirts, blue garments, and park images without containing any blue shirts in parks.

## Known limitations

- Environment labels are zero-shot predictions and require manual review, especially the 93 unknown images.
- Color is based on median masked pixels; multicolored and patterned garments are reduced to one dominant color.
- The subset is naturally skewed toward street fashion and shoes.
- Activity, style, and object annotations have not yet been generated.
- The subset is suitable for pipeline development, but targeted supplemental images are needed for balanced evaluation of all five assignment prompts.


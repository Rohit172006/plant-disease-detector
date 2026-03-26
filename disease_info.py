"""
PlantVillage Disease Information Database
==========================================
Contains disease descriptions, organic & inorganic treatments
for all 38 PlantVillage dataset classes.
"""

DISEASE_INFO = {
    # ── Apple ──────────────────────────────────────────────
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "description": "A fungal disease caused by Venturia inaequalis. Creates dark, scabby lesions on leaves and fruit, leading to premature defoliation.",
        "organic_treatment": "Apply neem oil spray weekly. Use compost tea as a foliar spray. Remove and destroy fallen infected leaves. Plant resistant varieties like Liberty or Enterprise.",
        "inorganic_treatment": "Apply captan or myclobutanil fungicide at bud break. Use mancozeb during growing season. Spray chlorothalonil preventively every 7-10 days.",
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "description": "Caused by the fungus Botryosphaeria obtusa. Creates brown lesions with concentric rings on leaves and causes fruit rot with black mummification.",
        "organic_treatment": "Prune dead or cankered wood promptly. Remove mummified fruits. Apply copper-based organic sprays during dormant season. Improve air circulation through proper pruning.",
        "inorganic_treatment": "Apply captan or thiophanate-methyl fungicide. Use myclobutanil during early growth stages. Spray mancozeb at petal fall and continue at 10-day intervals.",
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "description": "A fungal disease caused by Gymnosporangium juniperi-virginianae. Requires both apple and cedar/juniper trees to complete its lifecycle. Creates bright orange spots on leaves.",
        "organic_treatment": "Remove nearby cedar and juniper trees within 2 miles. Apply sulfur-based sprays in spring. Use resistant apple varieties like Freedom or Redfree.",
        "inorganic_treatment": "Apply myclobutanil or triadimefon at pink bud stage. Use propiconazole fungicide. Spray fenarimol at bloom and repeat every 10 days.",
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease or pest damage.",
        "organic_treatment": "Continue regular organic care: compost mulching, neem oil preventive sprays, and companion planting with marigolds.",
        "inorganic_treatment": "Continue regular care: balanced NPK fertilization, preventive fungicide schedule, and proper irrigation management.",
        "severity": "None"
    },

    # ── Blueberry ─────────────────────────────────────────
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Maintain acidic soil (pH 4.5-5.5) with pine needle mulch. Apply fish emulsion fertilizer. Use companion planting with thyme.",
        "inorganic_treatment": "Apply ammonium sulfate fertilizer for acidic conditions. Use balanced micronutrient sprays. Maintain proper irrigation schedule.",
        "severity": "None"
    },

    # ── Cherry ────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "description": "A fungal disease caused by Podosphaera clandestina. Creates white powdery coating on leaves, reducing photosynthesis and weakening the tree.",
        "organic_treatment": "Spray diluted milk solution (1:9 ratio). Apply potassium bicarbonate spray. Use neem oil weekly. Prune to improve air circulation.",
        "inorganic_treatment": "Apply sulfur-based fungicides early in the season. Use myclobutanil or trifloxystrobin. Spray propiconazole at first sign of infection.",
        "severity": "Moderate"
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Mulch with organic compost annually. Apply dormant oil spray in winter. Maintain proper tree spacing for airflow.",
        "inorganic_treatment": "Apply balanced fertilizer in early spring. Use preventive copper sprays during dormancy. Maintain consistent watering.",
        "severity": "None"
    },

    # ── Corn (Maize) ──────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "description": "Caused by Cercospora zeae-maydis. Creates rectangular gray-brown lesions between leaf veins, reducing photosynthesis and yield.",
        "organic_treatment": "Rotate crops (avoid corn-after-corn). Use resistant hybrids. Apply Bacillus subtilis-based biocontrol products. Incorporate crop residues to speed decomposition.",
        "inorganic_treatment": "Apply strobilurin fungicides (azoxystrobin) at early tassel. Use propiconazole or pyraclostrobin. Spray trifloxystrobin at VT/R1 growth stage.",
        "severity": "Moderate"
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "description": "Caused by Puccinia sorghi. Produces small, reddish-brown pustules on both leaf surfaces. Can reduce yields in susceptible varieties.",
        "organic_treatment": "Plant resistant hybrids. Rotate crops with non-cereal crops. Apply sulfur dust as preventive. Remove severely infected plant debris after harvest.",
        "inorganic_treatment": "Apply triazole fungicides (propiconazole) at early infection. Use azoxystrobin or pyraclostrobin. Spray mancozeb as preventive treatment.",
        "severity": "Moderate"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "description": "Caused by Exserohilum turcicum. Creates long, elliptical gray-green lesions on leaves that can coalesce and kill entire leaves.",
        "organic_treatment": "Use resistant hybrids (Ht genes). Practice crop rotation with soybeans or small grains. Till under crop residue. Apply Trichoderma-based biocontrol agents.",
        "inorganic_treatment": "Apply azoxystrobin or propiconazole at first sign. Use pyraclostrobin + metconazole combination. Spray chlorothalonil as protectant fungicide.",
        "severity": "High"
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Maintain crop rotation schedule. Apply compost side-dressing at knee-high stage. Use cover crops in off-season for soil health.",
        "inorganic_treatment": "Apply nitrogen fertilizer at recommended rates. Maintain proper plant spacing. Use balanced micronutrient program.",
        "severity": "None"
    },

    # ── Grape ─────────────────────────────────────────────
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "description": "Caused by Guignardia bidwellii. Creates circular tan spots on leaves and causes berries to shrivel into hard, black mummies.",
        "organic_treatment": "Remove mummified berries and infected canes. Apply copper-based organic sprays at early bloom. Improve canopy airflow through summer pruning.",
        "inorganic_treatment": "Apply myclobutanil at pre-bloom. Use mancozeb or captan at early shoot growth. Spray azoxystrobin at immediate pre-bloom and post-bloom.",
        "severity": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "description": "A complex fungal disease involving multiple pathogens. Causes tiger-stripe pattern on leaves and dark spots on berries. Can kill vines.",
        "organic_treatment": "Prune during dry weather to reduce infection. Apply Trichoderma-based biocontrol to pruning wounds. Remove severely infected vines. Avoid excessive vine stress.",
        "inorganic_treatment": "Apply thiophanate-methyl to pruning wounds. Use fosetyl-aluminum trunk injections. No fully effective chemical control exists; focus on prevention.",
        "severity": "High"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "description": "Creates irregular brown spots with dark borders on leaves. Severe infections cause premature defoliation affecting fruit quality.",
        "organic_treatment": "Remove infected leaves promptly. Apply copper hydroxide sprays. Improve vineyard drainage and air circulation. Use organic sulfur sprays preventively.",
        "inorganic_treatment": "Apply mancozeb or captan as protectant. Use carbendazim at first symptom appearance. Spray copper oxychloride at 15-day intervals.",
        "severity": "Moderate"
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Maintain proper canopy management. Apply compost tea foliar sprays. Use cover crops between rows for soil health.",
        "inorganic_treatment": "Apply balanced vine nutrition program. Use preventive fungicide schedule. Maintain drip irrigation for consistent soil moisture.",
        "severity": "None"
    },

    # ── Orange ────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "description": "A devastating bacterial disease spread by Asian citrus psyllid. Causes blotchy yellowing of leaves, lopsided fruit, and eventual tree death.",
        "organic_treatment": "Control psyllid vectors with organic insecticidal soaps and neem oil. Remove infected trees immediately. Use reflective mulch to deter psyllids. Apply nutritional sprays to support tree health.",
        "inorganic_treatment": "Apply systemic insecticides (imidacloprid) for psyllid control. Use foliar micronutrient sprays. No chemical cure exists; focus on vector management and tree removal.",
        "severity": "High"
    },

    # ── Peach ─────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "description": "Caused by Xanthomonas arboricola pv. pruni. Creates small, angular dark spots on leaves and sunken lesions on fruit.",
        "organic_treatment": "Plant resistant varieties. Apply copper hydroxide sprays at leaf fall. Use Bacillus-based biocontrol products. Avoid overhead irrigation to reduce leaf wetness.",
        "inorganic_treatment": "Apply oxytetracycline (Mycoshield) during bloom. Use copper + mancozeb tank mix. Spray streptomycin at early petal fall.",
        "severity": "Moderate"
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Apply dormant oil spray in late winter. Mulch with aged compost. Thin fruit to reduce disease pressure and stress.",
        "inorganic_treatment": "Apply balanced fertilizer in early spring. Use preventive fungicide at bud swell. Maintain proper irrigation schedule.",
        "severity": "None"
    },

    # ── Pepper ────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Pepper (Bell)",
        "disease": "Bacterial Spot",
        "description": "Caused by Xanthomonas species. Creates small, water-soaked spots on leaves that turn brown with yellow halos. Reduces fruit quality and yield.",
        "organic_treatment": "Use disease-free seeds and transplants. Apply copper-based organic sprays. Practice crop rotation (3-year cycle). Remove infected plant debris. Avoid overhead watering.",
        "inorganic_treatment": "Apply copper hydroxide + mancozeb tank mix weekly. Use acibenzolar-S-methyl (Actigard) for systemic resistance. Spray streptomycin at early infection.",
        "severity": "Moderate"
    },
    "Pepper,_bell___healthy": {
        "plant": "Pepper (Bell)",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Mulch with straw to retain moisture. Apply compost tea biweekly. Practice companion planting with basil to deter pests.",
        "inorganic_treatment": "Apply balanced NPK fertilizer. Use drip irrigation for consistent moisture. Apply preventive copper sprays during wet weather.",
        "severity": "None"
    },

    # ── Potato ────────────────────────────────────────────
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "description": "Caused by Alternaria solani. Creates dark brown spots with concentric rings (target spots) on older leaves first. Reduces tuber yield.",
        "organic_treatment": "Practice crop rotation (3+ years). Apply Bacillus subtilis biocontrol sprays. Use copper-based organic fungicides. Mulch to prevent soil splash. Remove infected foliage.",
        "inorganic_treatment": "Apply chlorothalonil or mancozeb at first symptom. Use azoxystrobin or difenoconazole. Spray boscalid + pyraclostrobin combination preventively.",
        "severity": "Moderate"
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "description": "Caused by Phytophthora infestans. Creates large, water-soaked lesions with white mold on leaf undersides. Extremely destructive; caused the Irish Potato Famine.",
        "organic_treatment": "Plant certified disease-free seed potatoes. Apply copper hydroxide sprays preventively. Destroy infected plants immediately. Hill soil around stems. Use resistant varieties like Sarpo Mira.",
        "inorganic_treatment": "Apply mancozeb + cymoxanil combination. Use dimethomorph or mandipropamid. Spray fluopicolide or cyazofamid for resistant strains. Apply every 5-7 days during outbreaks.",
        "severity": "High"
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Use certified seed potatoes. Apply compost at planting. Practice crop rotation with legumes. Mulch with straw after emergence.",
        "inorganic_treatment": "Apply balanced fertilizer at planting and side-dress at tuber initiation. Maintain consistent irrigation. Use preventive fungicide during humid conditions.",
        "severity": "None"
    },

    # ── Raspberry ─────────────────────────────────────────
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Prune old canes after fruiting. Apply aged compost mulch. Use drip irrigation to keep foliage dry.",
        "inorganic_treatment": "Apply balanced berry fertilizer in spring. Use preventive fungicide at bud break. Maintain proper cane spacing.",
        "severity": "None"
    },

    # ── Soybean ───────────────────────────────────────────
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Inoculate seeds with Bradyrhizobium for nitrogen fixation. Practice crop rotation. Use cover crops in off-season.",
        "inorganic_treatment": "Apply phosphorus and potassium based on soil test. Use seed treatment fungicides. Maintain proper plant population.",
        "severity": "None"
    },

    # ── Squash ────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "description": "Caused by Podosphaera xanthii and Erysiphe cichoracearum. Creates white powdery patches on leaves, reducing plant vigor and fruit production.",
        "organic_treatment": "Spray diluted milk (1:9) or baking soda solution biweekly. Apply neem oil. Plant resistant varieties. Ensure proper plant spacing for airflow.",
        "inorganic_treatment": "Apply myclobutanil or triadimefon at first sign. Use sulfur-based fungicides preventively. Spray trifloxystrobin or azoxystrobin every 7-14 days.",
        "severity": "Moderate"
    },

    # ── Strawberry ────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "description": "Caused by Diplocarpon earlianum. Creates irregular purple spots that merge, causing leaf margins to appear scorched. Weakens plants over time.",
        "organic_treatment": "Remove and destroy infected leaves. Apply neem oil sprays. Renovate beds after harvest by mowing foliage. Plant resistant varieties. Improve air circulation.",
        "inorganic_treatment": "Apply captan or myclobutanil fungicide at bloom. Use azoxystrobin at early fruit development. Spray thiophanate-methyl post-harvest.",
        "severity": "Moderate"
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "organic_treatment": "Mulch with straw to suppress weeds and retain moisture. Apply fish emulsion fertilizer monthly. Remove runners to maintain plant vigor.",
        "inorganic_treatment": "Apply balanced fertilizer at planting and after first harvest. Use drip irrigation. Apply preventive fungicide at bloom.",
        "severity": "None"
    },

    # ── Tomato ────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "description": "Caused by Xanthomonas species. Creates small, dark, water-soaked spots on leaves, stems, and fruit. Thrives in warm, wet conditions.",
        "organic_treatment": "Use disease-free seeds and transplants. Apply copper-based organic sprays early. Practice 3-year crop rotation. Remove infected plants. Avoid overhead irrigation.",
        "inorganic_treatment": "Apply copper hydroxide + mancozeb weekly during wet weather. Use acibenzolar-S-methyl for induced resistance. Spray fixed copper at transplanting.",
        "severity": "Moderate"
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description": "Caused by Alternaria solani. Creates dark concentric ring patterns (bull's-eye) on lower leaves first. Progresses upward causing defoliation.",
        "organic_treatment": "Mulch around plants to prevent soil splash. Apply Bacillus subtilis sprays. Remove lower infected leaves. Practice crop rotation. Use resistant varieties like Mountain Merit.",
        "inorganic_treatment": "Apply chlorothalonil at first symptom. Use azoxystrobin or difenoconazole. Spray mancozeb preventively every 7-10 days during humid weather.",
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description": "Caused by Phytophthora infestans. Creates large, dark, water-soaked patches with white fuzzy growth underneath. Can destroy entire crop rapidly.",
        "organic_treatment": "Remove and destroy infected plants immediately. Apply copper-based sprays preventively. Improve air circulation. Avoid overhead watering. Use resistant varieties like Defiant PHR.",
        "inorganic_treatment": "Apply chlorothalonil or mancozeb preventively. Use cymoxanil + famoxadone at first symptom. Spray mandipropamid or dimethomorph every 5-7 days during outbreaks.",
        "severity": "High"
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": "Caused by Passalora fulva (Cladosporium fulvum). Creates pale green-yellow spots on upper leaf surface with olive-green mold underneath. Thrives in high humidity.",
        "organic_treatment": "Improve greenhouse ventilation. Reduce humidity below 85%. Apply neem oil sprays. Remove infected lower leaves. Use resistant varieties with Cf resistance genes.",
        "inorganic_treatment": "Apply chlorothalonil or mancozeb preventively. Use azoxystrobin at first sign. Spray difenoconazole in greenhouse settings.",
        "severity": "Moderate"
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": "Caused by Septoria lycopersici. Creates numerous small, circular spots with dark borders and gray centers with tiny black dots (pycnidia).",
        "organic_treatment": "Remove infected lower leaves immediately. Apply copper-based organic fungicides. Mulch to prevent rain splash. Practice 3-year crop rotation. Stake plants for airflow.",
        "inorganic_treatment": "Apply chlorothalonil at first symptom and repeat every 7-10 days. Use mancozeb or azoxystrobin. Spray difenoconazole for severe infections.",
        "severity": "Moderate"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites (Two-Spotted)",
        "description": "Tiny arachnids (Tetranychus urticae) that feed on leaf undersides, causing stippling, yellowing, and fine webbing. Thrive in hot, dry conditions.",
        "organic_treatment": "Spray plants with strong water jet to dislodge mites. Apply neem oil or insecticidal soap weekly. Release predatory mites (Phytoseiulus persimilis). Use rosemary oil spray.",
        "inorganic_treatment": "Apply abamectin or bifenazate miticide. Use spiromesifen for resistant populations. Spray horticultural oil. Rotate miticide classes to prevent resistance.",
        "severity": "Moderate"
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "description": "Caused by Corynespora cassiicola. Creates brown spots with concentric rings and yellow halos on leaves, stems, and fruit.",
        "organic_treatment": "Remove and destroy infected plant material. Apply copper-based sprays. Improve plant spacing for airflow. Practice crop rotation with non-solanaceous crops.",
        "inorganic_treatment": "Apply chlorothalonil or mancozeb at first sign. Use azoxystrobin or difenoconazole. Spray boscalid + pyraclostrobin for severe cases.",
        "severity": "Moderate"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "description": "A devastating viral disease transmitted by whiteflies (Bemisia tabaci). Causes severe leaf curling, yellowing, stunting, and flower drop.",
        "organic_treatment": "Control whiteflies with yellow sticky traps and neem oil sprays. Use reflective silver mulch to repel whiteflies. Remove infected plants immediately. Plant resistant varieties like Ty hybrids.",
        "inorganic_treatment": "Apply imidacloprid or thiamethoxam systemic insecticides for whitefly control. Use pyriproxyfen for whitefly nymph control. No chemical cure for the virus; focus on vector management.",
        "severity": "High"
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "description": "A highly contagious viral disease causing mottled light and dark green mosaic patterns on leaves, stunted growth, and reduced fruit quality.",
        "organic_treatment": "Remove and destroy infected plants. Disinfect tools with 10% bleach solution. Use certified virus-free seeds. Wash hands before handling plants. Plant resistant varieties (Tm-2 gene).",
        "inorganic_treatment": "No chemical treatment available for viruses. Use virus-free seeds treated with trisodium phosphate. Apply systemic insecticides to control aphid vectors. Focus on sanitation and prevention.",
        "severity": "High"
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no visible signs of disease or pest damage.",
        "organic_treatment": "Mulch with organic material. Apply compost tea foliar sprays biweekly. Practice companion planting with basil and marigolds. Maintain consistent watering.",
        "inorganic_treatment": "Apply calcium-enriched fertilizer to prevent blossom end rot. Use preventive fungicide schedule during wet periods. Maintain drip irrigation.",
        "severity": "None"
    },
}

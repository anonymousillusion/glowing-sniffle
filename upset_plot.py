import pandas as pd
from matplotlib import pyplot as plt
from upsetplot import UpSet, from_memberships

# Define the generative techniques in each dataset
datasets = {
    "ILLUSION": {
        "MobileFaceSwap", "FSGAN", "FaceShifter", "ROOP", "DiffFace", "DiffSwap",
        "FreeVC", "XTTS", "DiffVC", "DiffHierVC", "YourTTS", "DiffGAN-TTS", "GradTTS",
        "Stable DiffusionXL", "Kandinsky 2.1", "MultiDiffusion", "SDXL-Turbo", 
        "This Person Does Not Exist", "AudioLDM", "MusicGen", "MAGNeT", "Audio Diffusion",
        "Text2Video-Zero", "ModelScopeT2V", "ZeroScope", "MidJourney", "ArtGuru", "Audalign"
    },
    "AGIQA-1K":{
        "stable-inpaintingv1 ", "SDv2"
    },
    "AV-Deepfake1M": {
      "VITS", "YourTTS", "TalkLip"  
    },
    "DF-Platter": {
        "FSGAN", "FaceShifter","FaceSwap"
        
    },
    "Midjourney-Kaggle": {
      "MidJourney"  
    },
    "LAV-DF":{
      "SV2TTS", "Wave2Lip"  
    },
    "DeePhy": {
        "FSGAN", "FaceShifter","FaceSwap"
    },
    "TIMIT-TTS": {
        "Tacotron", "Tacotron2", "GlowTTS","FastPitch", "FastSpeech2", "TalkNet",
        "MixerTTS", "MixerTTS-X", "VITS", "SpeedySpeech", "gTTS", "Silero", "MelGAN",
        "WaveRNN"
    },
    "FakeAVCeleb": {
      "SV2TTS", "FSGAN", "Wave2Lip", "FaceSwap"
    },
    
    "ForgeryNet": {
        "TalkingheadVideo", "FirstOrder Motion", "ATVG-Net", "MaskGAN", "SC-FEGAN",
        "StarGAN2", "StyleGAN2", "DiscoFace GAN", "BlendFace", "MM Replacement", "FaceSwap",
        "FSGAN", "FaceShifter", "SBS", "DSS"
    },
    "KoDF": {
        "FaceSwap", "DeepFaceLab", "FSGAN", " First Order Motion Model (FOMM)",
        "Audio-driven Talking Face Head Pose (ATFHP)", "Wave2Lip"
    },
    "DiffusionFace": {
        "P2","DDPM", "DDIM", "PNDM", "LDM", "SDv2.1 T2I", "SDv1.5 T2I", "SDv1.5 I2I",
        "SDv2.1 I2I", "inpaint", "DiffSwap"
    },
    "DIRE": {
        "DDPM", "iDDPM", "ADM", "PNDM", "LDM", "SDv2", "SDv1", "VQDiffusion"
    },
    "WaveFake": {
        "MelGAN", "Parallel WaveGAN (PWG)", "Multi-band MelGAN (MB-MelGAN)", 
        "Full-band MelGAN (FB-MelGAN)", "HiFi-GAN", "WaveGlow"
    }
} 

# Verify overlap directly between ILLUSION and DIRE
illusion_techniques = datasets["ILLUSION"]
dire_techniques = datasets["DIRE"]

# Compute intersection and size
intersection_illusion_dire = illusion_techniques & dire_techniques
print(f"Shared techniques between ILLUSION and DIRE: {intersection_illusion_dire}")
print(f"Number of shared techniques: {len(intersection_illusion_dire)}")


# Flatten the dataset memberships
memberships = []
for dataset, techniques_present in datasets.items():
    for technique in techniques_present:
        memberships.append((technique, dataset))

# Create a DataFrame
df = pd.DataFrame(memberships, columns=["Technique", "Dataset"])

# Group by Technique and collect Dataset memberships as a unique set
grouped = df.groupby("Technique")["Dataset"].apply(set).reset_index()

# Convert sets to sorted tuples for consistent group representation
grouped["Dataset"] = grouped["Dataset"].apply(lambda x: tuple(sorted(x)))

# Count occurrences of each unique group
grouped_counts = grouped["Dataset"].value_counts()

# Prepare data for UpSet plot
upset_data = from_memberships(grouped_counts.index, data=grouped_counts)

# Create the UpSet plot
upset_plot = UpSet(upset_data, intersection_plot_elements=10, show_counts=True)
upset_plot.plot()

# Show the plot
plt.title("Overlap of Generative Techniques Across Datasets")
plt.savefig("upset_plot.png")
# plt.show()


# Function to calculate Jaccard Index
def jaccard_index(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

# Compute Jaccard Index for each dataset pair with ILLUSION
jaccard_results = {}
for dataset_name, techniques in datasets.items():
    if dataset_name != "ILLUSION":
        jaccard_results[dataset_name] = jaccard_index(datasets["ILLUSION"], techniques)

# Display results
for dataset_name, jaccard in jaccard_results.items():
    print(f"Jaccard Index between ILLUSION and {dataset_name}: {jaccard:.2f}")
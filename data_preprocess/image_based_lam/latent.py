import tensorflow as tf
import os, sys, io
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from genie.modules import ControllableDINOLatentActionModel

# ============================================
# 1️⃣ provide TFRecord file path
# ============================================
if len(sys.argv) < 2:
    print("❌ please provide dataset path, for example: python latent.py /path/to/file.tfrecord")
    sys.exit(1)

tfrecord_path = sys.argv[1]
print(f"read file: {tfrecord_path}")

# ============================================
# 2️⃣ read TFRecord
# ============================================
dataset = tf.data.TFRecordDataset(tfrecord_path)

# ============================================
# 3️⃣ define feature description for parsing, based on features.json
# ============================================
feature_description = {
    "steps/is_first": tf.io.VarLenFeature(tf.int64),
    "steps/is_last": tf.io.VarLenFeature(tf.int64),
    "steps/is_terminal": tf.io.VarLenFeature(tf.int64),
    "steps/action": tf.io.VarLenFeature(tf.float32),
    "steps/discount": tf.io.VarLenFeature(tf.float32),
    "steps/reward": tf.io.VarLenFeature(tf.float32),
    "steps/observation/state": tf.io.VarLenFeature(tf.float32),
    "steps/observation/joint_state": tf.io.VarLenFeature(tf.float32),
    "steps/observation/image": tf.io.VarLenFeature(tf.string),
    "steps/observation/wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/language_instruction": tf.io.VarLenFeature(tf.string),
    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
}

def parse_example(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    for k, v in example.items():
        if isinstance(v, tf.SparseTensor):
            example[k] = tf.sparse.to_dense(v)
    return example


# ============================================
# 4️⃣ analysis trajectory
# ============================================
trajectories = []
for i, raw_record in enumerate(dataset):
    ex = parse_example(raw_record)
    num_steps = len(ex["steps/is_first"])

    # ---- define reshape ----
    action = ex["steps/action"].numpy().astype(np.float32).reshape(num_steps, 7)
    state = ex["steps/observation/state"].numpy().astype(np.float32).reshape(num_steps, 8)
    joint_state = ex["steps/observation/joint_state"].numpy().astype(np.float32).reshape(num_steps, 7)

    steps = []
    for t in range(num_steps):
        step_data = {
            "is_first": int(ex["steps/is_first"][t]),
            "is_last": int(ex["steps/is_last"][t]),
            "is_terminal": int(ex["steps/is_terminal"][t]),
            "action": action[t],           
            "reward": float(ex["steps/reward"][t]),
            "discount": float(ex["steps/discount"][t]),
            "state": state[t],            
            "joint_state": joint_state[t],
            "language_instruction": (
                ex["steps/language_instruction"].numpy()[0].decode("utf-8")
                if len(ex["steps/language_instruction"]) > 0 else ""
            ),
            "image": ex["steps/observation/image"].numpy()[t] if len(ex["steps/observation/image"]) > t else b"",
            "wrist_image": ex["steps/observation/wrist_image"].numpy()[t] if len(ex["steps/observation/wrist_image"]) > t else b"",
        }
        steps.append(step_data)

    trajectories.append(steps)
    print(f"✅ read trajectory {i}, total {num_steps} steps")

print(f"\n📊 file contains {len(trajectories)} trajectory")


# ============================================
# 5️⃣ load image-based LAM
# ============================================
lam_path = "your_lam_checkpoint.pth"  # TODO: provide your LAM checkpoint path here

def load_lam() -> ControllableDINOLatentActionModel:
    model = ControllableDINOLatentActionModel(
        in_dim=3, model_dim=768, latent_dim=128,
        num_latents=16, patch_size=14, enc_blocks=12,
        dec_blocks=12, num_heads=12, dropout=0.0,
    )
    ckpt = torch.load(lam_path, map_location="cpu")["state_dict"]
    ckpt = {k.replace("lam.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return model.eval()

latent_action_model = load_lam().to("cuda:0")
tokenizer = latent_action_model
to_tensor = transforms.ToTensor()


# ============================================
# 6️⃣ generate latent idx + z
# ============================================
all_latent_indices, all_latent_z = [], []

for traj_idx, traj in enumerate(tqdm(trajectories, desc="Processing trajectories")):
    traj_latent_idx, traj_latent_z = [], []
    num_steps = len(traj)

    for t in range(num_steps):
        img_bytes = traj[t]["image"]
        if not img_bytes:
            continue
        t_next = min(t + 11, num_steps - 1)
        img_next_bytes = traj[t_next]["image"]

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224), Image.BICUBIC)
        img_k = Image.open(io.BytesIO(img_next_bytes)).convert("RGB").resize((224, 224), Image.BICUBIC)

        video = torch.stack([to_tensor(img), to_tensor(img_k)], dim=0).unsqueeze(0).to("cuda:0")

        with torch.no_grad():
            vq_output = tokenizer.vq_encode(video)
            latent_idx = vq_output["indices"].squeeze().cpu().numpy().astype(np.float32)  # (4,)
            latent_z = vq_output["z"].squeeze().cpu().numpy().astype(np.float32)          # (4,128)

        traj_latent_idx.append(latent_idx)
        traj_latent_z.append(latent_z)

    all_latent_indices.append(traj_latent_idx)
    all_latent_z.append(traj_latent_z)
    print(f"✅ trajectroy {traj_idx} complete, total {len(traj_latent_idx)} steps")

print(f"\n🎯 total trajectory: {len(all_latent_indices)}.")


# ============================================
# 7️⃣ write new TFRecord dataset
# ============================================
output_dir = os.path.join(os.path.dirname(tfrecord_path), "output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, os.path.basename(tfrecord_path))
print(f"💾 begin new dataset writing: {output_path}")

def _bytes_feature(value):
    if isinstance(value, (list, np.ndarray)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.tobytes() if isinstance(v, np.ndarray) else v for v in value]))
    elif isinstance(value, bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value).encode()]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

with tf.io.TFRecordWriter(output_path) as writer:
    for traj_idx, traj in enumerate(trajectories):
        num_steps = len(traj)
        actions = np.stack([s["action"] for s in traj], axis=0).astype(np.float32)
        states = np.stack([s["state"] for s in traj], axis=0).astype(np.float32)
        joints = np.stack([s["joint_state"] for s in traj], axis=0).astype(np.float32)

        latents = np.stack(all_latent_indices[traj_idx], axis=0).astype(np.float32)
        z = np.stack(all_latent_z[traj_idx], axis=0).astype(np.float32)             

        feature = {
            "steps/is_first": _int64_feature([int(s["is_first"]) for s in traj]),
            "steps/is_last": _int64_feature([int(s["is_last"]) for s in traj]),
            "steps/is_terminal": _int64_feature([int(s["is_terminal"]) for s in traj]),
            "steps/action": _float_feature(actions.flatten().tolist()),
            "steps/discount": _float_feature([float(s["discount"]) for s in traj]),
            "steps/reward": _float_feature([float(s["reward"]) for s in traj]),
            "steps/observation/state": _float_feature(states.flatten().tolist()),
            "steps/observation/joint_state": _float_feature(joints.flatten().tolist()),
            "steps/language_instruction": _bytes_feature([s["language_instruction"].encode() for s in traj]),
            "steps/observation/image": _bytes_feature([s["image"] for s in traj]),
            "steps/observation/wrist_image": _bytes_feature([s["wrist_image"] for s in traj]),
            "episode_metadata/file_path": _bytes_feature([traj[0]["language_instruction"].encode() if traj[0]["language_instruction"] else b""]),
            "steps/latent_idx": _float_feature(latents.flatten().tolist()),
            "steps/latent_z": _float_feature(z.flatten().tolist()),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        print(f"✅ write trajectory {traj_idx} | steps={num_steps}, latent_z={z.shape}")

print(f"\n🎉 new dataset files save completly: {output_path}")

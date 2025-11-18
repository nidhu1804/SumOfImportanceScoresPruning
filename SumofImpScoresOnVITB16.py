import torch
from time import time
from abc import ABC, abstractmethod
import timm
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import pandas as pd
from tensorflow.io import gfile

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_timm(model_type, dataset_name, verbose=False, top10_idx=-1):
    """
    model   types: B/16, S/16 or Ti/16
    dataset names: cifar100 or oxford-iiit-pet
    top10_idx    : int [1, 10]
    Returns:
        model, res  -> res is int (224 or 384)
    """
    index = pd.read_csv('index.csv')
    pretrains = set(
        index.query('ds=="i21k"').groupby('name').apply(
            lambda df: df.sort_values('final_val').iloc[-1],
            include_groups=False).filename
    )
    finetunes = index.loc[index.filename.apply(lambda name: name in pretrains)]
    checkpoint = (
        finetunes.query(f'name=="{model_type}" and adapt_ds=="{dataset_name}"')
        .sort_values('adapt_final_val').iloc[-10].adapt_filename
    )  # example: Ti_16-i21k-300ep-...-cifar100-...-res_224
    if verbose: print(f"Loaded checkpoint: {checkpoint}")

    timm_modelnames = {
        'Ti/16-224': 'vit_tiny_patch16_224',
        'Ti/16-384': 'vit_tiny_patch16_384',
        'S/16-224': 'vit_small_patch16_224',
        'S/16-384': 'vit_small_patch16_384',
        'B/16-224': 'vit_base_patch16_224',
        'B/16-384': 'vit_base_patch16_384'
    }
    num_classes = 100 if dataset_name == 'cifar100' else 37
    res = int(checkpoint.split('_')[-1])
    model = timm.create_model(timm_modelnames[f'{model_type}-{res}'], num_classes=num_classes)

    # downloading a checkpoint automatically
    # may show an error, but still downloads the checkpoint
    if not gfile.exists(f'models/{checkpoint}.npz'):
        gfile.copy(f'gs://vit_models/augreg/{checkpoint}.npz', f'models/{checkpoint}.npz')
    timm.models.load_checkpoint(model, f'models/{checkpoint}.npz')

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, res


def seed_worker(worker_id):
    worker_seed = seed % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataset(dataset_name, batch_size, model_cfg=None, subset_size=1., res=(224, 224), train=False,
                 download_dataset=False, do_balanced_subset=False):

    dataset = (
        datasets.CIFAR100('data/', train=train, download=download_dataset) if dataset_name == 'cifar100'
        else datasets.OxfordIIITPet('data/', split=('trainval' if train else 'test'))
    )
    if model_cfg is None:
        m, s = [0.5] * 3, [0.5] * 3
    else:
        m, s = model_cfg['mean'], model_cfg['std']
        res = model_cfg['input_size'][-2:]
    dataset.transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=m, std=s),
        v2.Resize(res),
    ])
    g = torch.Generator()
    g.manual_seed(seed)
    if subset_size < 1.0:
        n = len(dataset)
        n_small = int(subset_size * n)
        dataset, _ = random_split(dataset, [n_small, n - n_small], generator=g)
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, worker_init_fn=seed_worker, generator=g)
    return dataset, dataloader

class ViTModelAdapter:
    """Adapter for timm Vision Transformer models"""

    def __init__(self, vit_model):

        self.model = vit_model

        # Extract model dimensions
        self.de = vit_model.embed_dim
        self.nh = vit_model.blocks[0].attn.num_heads
        self.dh = vit_model.embed_dim // self.nh
        self.nb = len(vit_model.blocks)

        # Create block adapters
        self.bs = []
        for block in vit_model.blocks:
            self.bs.append(ViTBlockAdapter(block, self.de, self.nh, self.dh))

    def count_parameters(self):
        """Count total parameters in attention and MLP"""
        total = 0
        for block in self.bs:
            total += block.q.numel() + block.k.numel() + block.v.numel() + block.p.numel()
            total += block.fc1.numel() + block.fc2.numel()
        return total

    def count_nonzero_parameters(self):
        """Count non-zero parameters"""
        nonzero = 0
        for block in self.bs:
            nonzero += (block.q != 0).sum().item()
            nonzero += (block.k != 0).sum().item()
            nonzero += (block.v != 0).sum().item()
            nonzero += (block.p != 0).sum().item()
            nonzero += (block.fc1 != 0).sum().item()
            nonzero += (block.fc2 != 0).sum().item()
        return nonzero


class ViTBlockAdapter:
    """Adapter for individual ViT blocks"""

    def __init__(self, block, emb_dim, n_heads, head_dim):
        """Extract weight matrices from timm ViT block"""
        # Extract attention weights
        qkv_weight = block.attn.qkv.weight.data

        # Split into q, k, v
        self.q = qkv_weight[:emb_dim, :].clone()
        self.k = qkv_weight[emb_dim:2 * emb_dim, :].clone()
        self.v = qkv_weight[2 * emb_dim:, :].clone()

        # Extract projection weight
        self.p = block.attn.proj.weight.data.clone()

        # Extract MLP weights
        self.fc1 = block.mlp.fc1.weight.data.clone()
        self.fc2 = block.mlp.fc2.weight.data.clone()


# Evaluation Functions

@torch.no_grad()
def evaluate_accuracy(model, test_loader, device):
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def measure_latency(model, device, input_res=224, num_iterations=100, warmup=10):
    """Measure inference latency (returns milliseconds)"""
    model.eval()
    dummy_input = torch.randn((1, 3, input_res, input_res)).to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input)

    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    start = time()
    for _ in range(num_iterations):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end = time()
    latency_ms = (end - start) / num_iterations * 1000  # Convert to milliseconds

    return latency_ms


# PRUNING METHOD INTERFACE
class PruningMethod(ABC):
    """
    UNIFIED INTERFACE for plugging in ANY pruning method

    FLEXIBLE ARCHITECTURE:
    - You can provide ONLY mlp_neuron_importance (MLP-only pruning)
    - You can provide ONLY att_importance (Attention-only pruning)
    - You can provide BOTH (prune both components)
    """

    def __init__(self, model, attention_granularity=None, dataloader=None):
        self.model = model
        self.attention_granularity = attention_granularity
        self.dataloader = dataloader

        if attention_granularity is not None:
            attention_granularity = attention_granularity.lower()
            if attention_granularity not in ['width', 'depth', 'head']:
                raise ValueError(f"Invalid attention_granularity: {attention_granularity}")

        self.mlp_neuron_importance = None
        self.att_importance = None

    @abstractmethod
    def compute_importance_scores(self):
        pass

    def get_mlp_importance(self):
        if self.mlp_neuron_importance is None:
            self.compute_importance_scores()
        return self.mlp_neuron_importance

    def get_att_importance(self):
        if self.att_importance is None:
            self.compute_importance_scores()
        return self.att_importance


class MLP_L1Method(PruningMethod):
    """Prunes ONLY MLP using L1 norm"""

    def __init__(self, model, dataloader=None):
        super().__init__(model, attention_granularity=None, dataloader=dataloader)

    def compute_importance_scores(self):
        self.mlp_neuron_importance = []

        for block in self.model.bs:
            mlp_scores = torch.sum(torch.abs(block.fc1), dim=1)
            mlp_scores += torch.sum(torch.abs(block.fc2), dim=0)
            mlp_scores = mlp_scores / (mlp_scores.max() + 1e-8)
            self.mlp_neuron_importance.append(mlp_scores)

        self.att_importance = None


# Example Implementations
class Attention_HeadMethod(PruningMethod):
    """Prunes ONLY Attention at head-level"""

    def __init__(self, model, dataloader=None):
        super().__init__(model, attention_granularity='head', dataloader=dataloader)

    def compute_importance_scores(self):
        self.att_importance = []

        nh = self.model.nh
        dh = self.model.dh

        for block in self.model.bs:
            head_scores = torch.zeros(nh)

            for h in range(nh):
                start = h * dh
                end = (h + 1) * dh

                score = torch.abs(block.q[start:end]).sum()
                score += torch.abs(block.k[start:end]).sum()
                score += torch.abs(block.v[start:end]).sum()
                score += torch.abs(block.p[:, start:end]).sum()

                head_scores[h] = score.item()

            head_scores = head_scores / (head_scores.max() + 1e-8)
            self.att_importance.append(head_scores)

        self.mlp_neuron_importance = None


class MLP_L2Method(PruningMethod):
    """Prunes ONLY MLP using L2 norm"""

    def __init__(self, model, dataloader=None):
        super().__init__(model, attention_granularity=None, dataloader=dataloader)

    def compute_importance_scores(self):
        self.mlp_neuron_importance = []

        for block in self.model.bs:
            mlp_scores = torch.sum(block.fc1 ** 2, dim=1)
            mlp_scores += torch.sum(block.fc2 ** 2, dim=0)
            mlp_scores = mlp_scores / (mlp_scores.max() + 1e-8)
            self.mlp_neuron_importance.append(mlp_scores)

        self.att_importance = None


class Attention_WidthMethod(PruningMethod):
    """Prunes ONLY Attention at neuron-level"""

    def __init__(self, model, dataloader=None):
        super().__init__(model, attention_granularity='width', dataloader=dataloader)

    def compute_importance_scores(self):
        self.att_importance = []

        for block in self.model.bs:
            att_scores = torch.sum(torch.abs(block.q), dim=1)
            att_scores += torch.sum(torch.abs(block.k), dim=1)
            att_scores += torch.sum(torch.abs(block.v), dim=1)
            att_scores += torch.sum(torch.abs(block.p), dim=0)
            att_scores = att_scores / (att_scores.max() + 1e-8)
            self.att_importance.append(att_scores)

        self.mlp_neuron_importance = None


class SumOfImportanceScoresPruning:
    """Combines multiple pruning methods flexibly"""

    def __init__(self, model, pruning_methods):
        self.model = model
        self.methods = pruning_methods

        self.mlp_methods = []
        self.att_methods = []

        for method in self.methods:
            method.compute_importance_scores()
            if method.mlp_neuron_importance is not None:
                self.mlp_methods.append(method)
            if method.att_importance is not None:
                self.att_methods.append(method)

        print(f"  → {len(self.mlp_methods)} methods pruning MLP")
        print(f"  → {len(self.att_methods)} methods pruning Attention")

    def _expand_att_scores_to_neurons(self, method, block_idx):
        if method.attention_granularity == 'width':
            return method.get_att_importance()[block_idx]
        elif method.attention_granularity == 'depth':
            block_score = method.get_att_importance()[block_idx].item()
            return torch.full((self.model.nh * self.model.dh,), block_score)
        elif method.attention_granularity == 'head':
            head_scores = method.get_att_importance()[block_idx]
            neuron_scores = torch.zeros(self.model.nh * self.model.dh)
            for h in range(self.model.nh):
                start = h * self.model.dh
                end = (h + 1) * self.model.dh
                neuron_scores[start:end] = head_scores[h]
            return neuron_scores

    def combine_scores(self):
        nb = self.model.nb
        mlp_hidden_dim = self.model.bs[0].fc1.shape[0]
        att_hidden_dim = self.model.nh * self.model.dh

        combined_mlp_scores = None
        combined_att_scores = None

        if self.mlp_methods:
            combined_mlp_scores = [torch.zeros(mlp_hidden_dim) for _ in range(nb)]
            for method in self.mlp_methods:
                mlp_importance = method.get_mlp_importance()
                for block_idx in range(nb):
                    combined_mlp_scores[block_idx] += mlp_importance[block_idx]

        if self.att_methods:
            combined_att_scores = [torch.zeros(att_hidden_dim) for _ in range(nb)]
            for method in self.att_methods:
                for block_idx in range(nb):
                    att_neuron_scores = self._expand_att_scores_to_neurons(method, block_idx)
                    combined_att_scores[block_idx] += att_neuron_scores

        return combined_mlp_scores, combined_att_scores

    def create_masks(self, target_sparsity):
        combined_mlp, combined_att = self.combine_scores()

        mlp_masks = None
        att_masks = None

        if combined_mlp is not None:
            mlp_masks = self._create_mlp_masks(combined_mlp, target_sparsity)

        if combined_att is not None:
            att_masks = self._create_att_masks(combined_att, target_sparsity)

        return mlp_masks, att_masks

    def _create_mlp_masks(self, combined_scores, target_sparsity):
        all_scores = torch.cat(combined_scores)
        sorted_scores, _ = torch.sort(all_scores)
        threshold_idx = int(target_sparsity * len(sorted_scores))
        threshold = sorted_scores[threshold_idx].item()

        masks = []
        for i, block in enumerate(self.model.bs):
            neuron_mask = (combined_scores[i] > threshold).float()
            fc1_mask = neuron_mask.unsqueeze(1).expand_as(block.fc1)
            fc2_mask = neuron_mask.unsqueeze(0).expand_as(block.fc2)
            masks.append([fc1_mask, fc2_mask])

        return masks

    def _create_att_masks(self, combined_scores, target_sparsity):
        all_scores = torch.cat(combined_scores)
        sorted_scores, _ = torch.sort(all_scores)
        threshold_idx = int(target_sparsity * len(sorted_scores))
        threshold = sorted_scores[threshold_idx].item()

        masks = []
        for i, block in enumerate(self.model.bs):
            neuron_mask = (combined_scores[i] > threshold).float()
            q_mask = neuron_mask.unsqueeze(1).expand_as(block.q)
            k_mask = neuron_mask.unsqueeze(1).expand_as(block.k)
            v_mask = neuron_mask.unsqueeze(1).expand_as(block.v)
            p_mask = neuron_mask.unsqueeze(0).expand_as(block.p)
            masks.append([q_mask, k_mask, v_mask, p_mask])

        return masks

    def apply_masks_to_model(self, vit_model, mlp_masks, att_masks):
        """Apply masks directly to the original timm model"""
        if att_masks is not None:
            for i, block in enumerate(vit_model.blocks):
                # Apply to QKV (combined weight)
                qkv_mask = torch.cat([
                    att_masks[i][0],  # Q mask
                    att_masks[i][1],  # K mask
                    att_masks[i][2],  # V mask
                ], dim=0)
                block.attn.qkv.weight.data.mul_(qkv_mask)

                # Apply to projection
                block.attn.proj.weight.data.mul_(att_masks[i][3])

        if mlp_masks is not None:
            for i, block in enumerate(vit_model.blocks):
                block.mlp.fc1.weight.data.mul_(mlp_masks[i][0])
                block.mlp.fc2.weight.data.mul_(mlp_masks[i][1])

    def get_sparsity_stats(self, mlp_masks, att_masks):
        mlp_sparsity = 0.0
        att_sparsity = 0.0

        if mlp_masks is not None:
            mlp_total = sum(m[0].numel() + m[1].numel() for m in mlp_masks)
            mlp_pruned = sum((m[0] == 0).sum() + (m[1] == 0).sum() for m in mlp_masks)
            mlp_sparsity = mlp_pruned.item() / mlp_total

        if att_masks is not None:
            att_total = sum(sum(m.numel() for m in block) for block in att_masks)
            att_pruned = sum(sum((m == 0).sum() for m in block) for block in att_masks)
            att_sparsity = att_pruned.item() / att_total

        total_params = self.model.count_parameters()
        active_params = self.model.count_nonzero_parameters()
        total_sparsity = 1 - (active_params / total_params)

        return mlp_sparsity, att_sparsity, total_sparsity


def run_experiment(model, test_loader, device, target_sparsity, pruning_methods, experiment_name, input_res):
    """Run single pruning experiment with full evaluation"""

    print(f"\n{'=' * 80}")
    print(f"Experiment: {experiment_name}")
    print(f"Target Sparsity: {target_sparsity * 100:.1f}%")
    print(f"{'=' * 80}")

    # Measure baseline
    print("\n📊 Baseline Model Performance:")
    base_acc = evaluate_accuracy(model, test_loader, device)
    base_latency = measure_latency(model, device, input_res)
    print(f"  Accuracy: {base_acc:.2f}%")
    print(f"  Latency:  {base_latency:.2f} ms")

    # Create adapter and pruner
    print("\n🔧 Creating pruning masks...")
    adapted_model = ViTModelAdapter(model)

    start = time()
    pruner = SumOfImportanceScoresPruning(adapted_model, pruning_methods)
    mlp_masks, att_masks = pruner.create_masks(target_sparsity)
    mask_time = time() - start

    # Get sparsity stats
    mlp_sp, att_sp, total_sp = pruner.get_sparsity_stats(mlp_masks, att_masks)

    # Apply masks to original model
    print("\n✂️  Applying pruning masks...")
    pruner.apply_masks_to_model(model, mlp_masks, att_masks)

    # Measure after pruning
    print("\n📊 Pruned Model Performance:")
    pruned_acc = evaluate_accuracy(model, test_loader, device)
    pruned_latency = measure_latency(model, device, input_res)
    print(f"  Accuracy: {pruned_acc:.2f}%")
    print(f"  Latency:  {pruned_latency:.2f} ms")

    # Calculate improvements
    acc_drop = base_acc - pruned_acc
    latency_reduction = ((base_latency - pruned_latency) / base_latency) * 100 if base_latency > 0 else 0.0

    print(f"\n📈 Results Summary:")
    print(f"  MLP Sparsity:       {mlp_sp * 100:5.1f}%")
    print(f"  Attention Sparsity: {att_sp * 100:5.1f}%")
    print(f"  Total Sparsity:     {total_sp * 100:5.1f}%")
    print(f"  Accuracy Drop:      {acc_drop:5.2f}%")
    print(f"  Latency Reduction:  {latency_reduction:5.1f}%")
    print(f"  Mask Creation Time: {mask_time:.3f}s")

    return {
        'name': experiment_name,
        'base_acc': base_acc,
        'pruned_acc': pruned_acc,
        'acc_drop': acc_drop,
        'base_latency': base_latency,
        'pruned_latency': pruned_latency,
        'latency_reduction': latency_reduction,
        'mlp_sp': mlp_sp,
        'att_sp': att_sp,
        'total_sp': total_sp,
        'time': mask_time
    }


def main():

    print(f"Using device: {device}")

    # Configuration
    model_type = 'B/16'  # Options: 'Ti/16', 'S/16', 'B/16'
    dataset_name = 'cifar100'
    batch_size = 128
    target_sparsity = 0.3  # 30% pruning

    # Load model (capture resolution)
    print(f"\n📦 Loading {model_type} model for {dataset_name}...")
    model, input_res = load_model_timm(model_type, dataset_name, verbose=True)
    model_cfg = model.default_cfg

    # Load dataset
    print(f"\n📦 Loading {dataset_name} dataset...")
    _, test_loader = load_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        model_cfg=model_cfg,
        train=False,
        download_dataset=True
    )
    print(f"  Test samples: {len(test_loader.dataset)}")

    results = []

    print("\n" + "=" * 80)
    print(" " * 25 + "STARTING EXPERIMENTS")
    print("=" * 80)

    # Experiment 1: MLP-only pruning (L1)
    print("\n" + "▶" * 40)
    print("EXPERIMENT 1: MLP-Only Pruning (L1)")
    print("▶" * 40)

    model, input_res = load_model_timm(model_type, dataset_name, verbose=False)
    adapted = ViTModelAdapter(model)
    methods = [MLP_L1Method(adapted)]
    results.append(run_experiment(model, test_loader, device, target_sparsity, methods, "MLP-Only (L1)", input_res))

    # Experiment 2: Attention-only pruning (Head)
    print("\n" + "▶" * 40)
    print("EXPERIMENT 2: Attention-Only Pruning (Head)")
    print("▶" * 40)

    model, input_res = load_model_timm(model_type, dataset_name, verbose=False)
    adapted = ViTModelAdapter(model)
    methods = [Attention_HeadMethod(adapted)]
    results.append(run_experiment(model, test_loader, device, target_sparsity, methods, "Attention-Only (Head)", input_res))

    # Experiment 3: Combined pruning (MLP-L1 + Attention-Head)
    print("\n" + "▶" * 40)
    print("EXPERIMENT 3: Combined Pruning (MLP + Attention)")
    print("▶" * 40)

    model, input_res = load_model_timm(model_type, dataset_name, verbose=False)
    adapted = ViTModelAdapter(model)
    methods = [MLP_L1Method(adapted), Attention_HeadMethod(adapted)]
    results.append(run_experiment(model, test_loader, device, target_sparsity, methods, "MLP-L1 + Att-Head", input_res))

    # Experiment 4: Combined (MLP-L2 + Attention-Width)
    print("\n" + "▶" * 40)
    print("EXPERIMENT 4: Combined Pruning (MLP-L2 + Att-Width)")
    print("▶" * 40)

    model, input_res = load_model_timm(model_type, dataset_name, verbose=False)
    adapted = ViTModelAdapter(model)
    methods = [MLP_L2Method(adapted), Attention_WidthMethod(adapted)]
    results.append(run_experiment(model, test_loader, device, target_sparsity, methods, "MLP-L2 + Att-Width", input_res))

    # Print final summary table
    print("\n" + "=" * 80)
    print(" " * 25 + "FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Model: {model_type} | Dataset: {dataset_name} | Target Sparsity: {target_sparsity * 100:.0f}%")
    print("-" * 80)
    print(f"{'Method':<25} {'Base Acc':<10} {'Pruned Acc':<12} {'Acc Drop':<10} "
          f"{'Base Lat':<10} {'Pruned Lat':<12} {'Lat↓%':<8}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<25} {r['base_acc']:6.2f}%    {r['pruned_acc']:6.2f}%      "
              f"{r['acc_drop']:5.2f}%    {r['base_latency']:6.2f}ms   "
              f"{r['pruned_latency']:6.2f}ms      {r['latency_reduction']:5.1f}%")

    print("=" * 80)

    print("\n✅ All experiments completed successfully!")
    print(f"   Total experiments: {len(results)}")
    print(f"   Device used: {device}")
    print(f"   Model type: {model_type}")


if __name__ == "__main__":
    main()

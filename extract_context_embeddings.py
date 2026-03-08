#!/usr/bin/env python
import argparse
import copy
import os
import random

import numpy as np
import torch

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize, gather_nodes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract context-only per-position ProteinMPNN embeddings (encoder states). "
            "Output has shape (L, nd), where nd is model hidden dim."
        )
    )
    parser.add_argument("--pdb_path", type=str, required=True, help="Input PDB path")
    parser.add_argument(
        "--pdb_path_chains",
        type=str,
        default="",
        help="Optional space-separated chains, e.g. 'A B C D'. If empty, all chains are used.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="Output .npz path for embeddings and metadata",
    )
    parser.add_argument(
        "--path_to_model_weights",
        type=str,
        default="",
        help="Path to model weights folder. If empty, uses bundled weights.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="v_48_020",
        help="Model checkpoint stem, e.g. v_48_020",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="encoder",
        choices=["encoder", "aa_log_probs_20", "aa_probs_20"],
        help=(
            "Output representation: "
            "'encoder' -> context encoder states (L, hidden_dim); "
            "'aa_log_probs_20' -> context-only log-probabilities over 20 AAs (L, 20); "
            "'aa_probs_20' -> context-only probabilities over 20 AAs (L, 20)."
        ),
    )
    parser.add_argument("--ca_only", action="store_true", default=False)
    parser.add_argument("--use_soluble_model", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def get_model_folder(args):
    if args.path_to_model_weights:
        model_folder_path = args.path_to_model_weights
        if model_folder_path[-1] != "/":
            model_folder_path = model_folder_path + "/"
        return model_folder_path

    file_path = os.path.realpath(__file__)
    k = file_path.rfind("/")
    root = file_path[:k].replace("/helper_scripts", "")

    if args.ca_only:
        if args.use_soluble_model:
            raise ValueError("CA + soluble model combination is not available.")
        return root + "/ca_model_weights/"

    if args.use_soluble_model:
        return root + "/soluble_model_weights/"
    return root + "/vanilla_model_weights/"


def main():
    args = parse_args()

    seed = args.seed if args.seed else int(np.random.randint(0, high=999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_dim = 128
    num_layers = 3

    model_folder_path = get_model_folder(args)
    checkpoint_path = model_folder_path + f"{args.model_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ProteinMPNN(
        ca_only=args.ca_only,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    pdb_dict_list = parse_PDB(args.pdb_path, ca_only=args.ca_only)
    if len(pdb_dict_list) == 0:
        raise ValueError(f"No parseable chains found in PDB: {args.pdb_path}")
    protein = pdb_dict_list[0]

    all_chain_list = [item[-1:] for item in list(protein) if item[:9] == "seq_chain"]
    if args.pdb_path_chains:
        selected_chains = [str(item) for item in args.pdb_path_chains.split()]
    else:
        selected_chains = all_chain_list

    fixed_chains = [letter for letter in all_chain_list if letter not in selected_chains]
    chain_id_dict = {protein["name"]: (selected_chains, fixed_chains)}

    batch_clones = [copy.deepcopy(protein)]
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch_clones,
        device,
        chain_id_dict,
        fixed_position_dict=None,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None,
        ca_only=args.ca_only,
    )

    with torch.no_grad():
        E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = model.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        log_probs_ctx = None
        if args.output_type in ["aa_log_probs_20", "aa_probs_20"]:
            log_probs_ctx = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)

    mask0 = mask[0].bool()
    if args.output_type == "encoder":
        embeddings = h_V[0][mask0].detach().cpu().numpy().astype(np.float32)
    elif args.output_type == "aa_log_probs_20":
        embeddings = log_probs_ctx[0][mask0, :20].detach().cpu().numpy().astype(np.float32)
    else:
        embeddings = torch.exp(log_probs_ctx[0][mask0, :20]).detach().cpu().numpy().astype(np.float32)

    residue_idx_out = residue_idx[0][mask0].detach().cpu().numpy().astype(np.int32)
    chain_encoding_out = chain_encoding_all[0][mask0].detach().cpu().numpy().astype(np.int32)
    seq_token_idx_out = S[0][mask0].detach().cpu().numpy().astype(np.int32)

    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWYX"))
    aa_alphabet_20 = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    seq_chars_out = alphabet[seq_token_idx_out]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    np.savez(
        args.out_file,
        embeddings=embeddings,
        residue_idx=residue_idx_out,
        chain_encoding=chain_encoding_out,
        seq_token_idx=seq_token_idx_out,
        seq_chars=seq_chars_out,
        aa_alphabet_20=aa_alphabet_20,
        output_type=np.array(args.output_type),
        pdb_path=np.array(args.pdb_path),
        selected_chains=np.array(selected_chains, dtype=object),
        model_name=np.array(args.model_name),
        k_neighbors=np.array(int(checkpoint["num_edges"])),
        hidden_dim=np.array(int(embeddings.shape[1])),
        length=np.array(int(embeddings.shape[0])),
    )

    print(f"Saved embeddings to: {args.out_file}")
    print(f"embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()

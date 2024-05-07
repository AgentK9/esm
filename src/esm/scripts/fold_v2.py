from json import dumps
from pathlib import Path
from string import ascii_uppercase, ascii_lowercase
import hashlib
from typing import Literal

import numpy as np
import torch
import click
from scipy.special import softmax

from esm.scripts.fold import create_batched_sequence_datasest


class FASTAParseError(Exception):
    pass


def parse_output(output):
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0, :, 1]

    bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
    sm_contacts = softmax(output["distogram_logits"], -1)[0]
    sm_contacts = sm_contacts[..., bins < 8].sum(-1)
    xyz = output["positions"][-1, 0, :, 1]
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    o = {
        "pae": pae[mask, :][:, mask],
        "plddt": plddt[mask],
        "sm_contacts": sm_contacts[mask, :][:, mask],
        "xyz": xyz[mask],
    }
    return o


def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()


alphabet_list = list(ascii_uppercase + ascii_lowercase)


@click.command()
@click.version_option()
@click.option(
    "--sequence-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the sequence file",
    required=True,
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the model file",
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    help="Path to the output dir",
    required=True,
)
@click.option(
    "--processor",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to use for computation",
    default="cuda",
    show_default=True,
)
def main(sequence_path: Path, model_path: Path, output_dir: Path, processor: Literal["cuda", "cpu"]):
    all_sequences = {}
    header = None
    seq = None
    for line in sequence_path.read_text().splitlines():
        if not line:
            continue
        if line[0] == ">":
            header = line[1:].strip()
            seq = ""
        else:
            if not header:
                raise FASTAParseError(f"FASTA file {sequence_path} does not start with a header")
            seq += line
            all_sequences[header] = seq

    model = torch.load(str(model_path))
    model.eval()
    model.set_chunk_size(None)
    if processor == "cuda":
        model.cuda()
    elif processor == "cpu":
        model.esm.float()
        model.cpu()
    else:
        raise ValueError(f"Invalid processor {processor}")
    batched_sequences = create_batched_sequence_datasest(all_sequences)

    num_completed = 0
    for i, (headers, sequences) in enumerate(batched_sequences):
        try:
            output = model.infer(sequences)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                continue
            raise

        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        for j, (header, seq, pdb_string, mean_plddt, ptm) in enumerate(
            zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
            )
        ):
            output_file = output_dir / f"{i*j}.pdb"
            output_file.write_text(pdb_string)
            output_file.with_suffix(".metadata.json").write_text(
                dumps(
                    {
                        "plddt": mean_plddt.item(),
                        "ptm": ptm.item(),
                    }
                )
            )
            num_completed += 1


if __name__ == "__main__":
    main()

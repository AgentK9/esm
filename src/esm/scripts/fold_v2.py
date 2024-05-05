from pathlib import Path
from string import ascii_uppercase, ascii_lowercase
import hashlib
import re
import numpy as np
import torch
from scipy.special import softmax

from esm.scripts.fold import create_batched_sequence_datasest


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


def main():
    sequence = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/", ":").upper())
    sequence = re.sub(":+", ":", sequence)
    sequence = re.sub("^[:]+", "", sequence)
    sequence = re.sub("[:]+$", "", sequence)
    copies = 1
    sequence = ":".join([sequence] * copies)

    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    print("length", length)

    model_path = Path(__file__).parent.parent.parent.parent / "working" / "esmfold.model"

    model = torch.load(str(model_path))
    model.esm.float()
    model.eval()
    model.set_chunk_size(None)
    model.cpu()
    batched_sequences = create_batched_sequence_datasest([("test", sequence)])

    num_completed = 0
    for headers, sequences in batched_sequences:
        try:
            output = model.infer(sequences)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                continue
            raise

        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        for header, seq, pdb_string, mean_plddt, ptm in zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            output_file = Path(f"{header}.pdb")
            output_file.write_text(pdb_string)
            num_completed += 1


if __name__ == "__main__":
    main()

"""Command-line entry point for HLoc visualization workflows."""

import argparse
import re
from pathlib import Path

from .io_utils import discover, discover_local_features, h5_image_names, resolve_image
from .matches import render_matches
from .retrieval import read_pairs, rank_from_descriptors, render_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HLoc global retrieval and local feature matches.")
    parser.add_argument("--query", type=Path, default=Path(r"D:\Datasets\Indoor_query_38_38"), help="Query dataset root")
    parser.add_argument("--map", type=Path, default=Path(r"D:\Datasets\Indoor_map_51"), help="Map dataset root")
    parser.add_argument("--retrieval", choices=("netvlad", "megaloc"), default="netvlad", help="Global retrieval method")
    parser.add_argument("--matcher", default="auto", help="Matcher tag in matches HDF5 filename, or 'auto'")
    parser.add_argument("--top-k", type=int, default=10, help="Number of map images to visualize per query")
    parser.add_argument("--output", type=Path, default=Path("visualizations"), help="Output directory")
    parser.add_argument("--pairs", type=Path, help="Override retrieval pairs text file")
    parser.add_argument("--query-global", type=Path, help="Override query global descriptor HDF5")
    parser.add_argument("--map-global", type=Path, help="Override map global descriptor HDF5")
    parser.add_argument("--matches", type=Path, help="Override local matches HDF5")
    parser.add_argument("--query-features", type=Path, help="Override query local features HDF5")
    parser.add_argument("--map-features", type=Path, help="Override map local features HDF5")
    parser.add_argument("--max-queries", type=int, default=0, help="Limit query count; zero means all")
    parser.add_argument("--max-local-pairs", type=int, default=0, help="Limit match images; zero means all available pairs")
    parser.add_argument("--ransac-threshold", type=float, default=3.0, help="Fundamental-matrix RANSAC pixel threshold")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return args


def selected_rankings(args: argparse.Namespace, query_out: Path, map_out: Path):
    """Prefer the chosen method's HLoc pairs file, otherwise rank descriptors."""
    query_global = args.query_global or discover(query_out, f"global-feats-{args.retrieval}*.h5", f"{args.retrieval} query descriptors")
    map_global = args.map_global or discover(map_out, f"global-feats-{args.retrieval}*.h5", f"{args.retrieval} map descriptors")
    query_names, map_names = h5_image_names(query_global), h5_image_names(map_global)
    pairs_file = args.pairs
    if pairs_file is None:
        candidates = sorted(query_out.glob(f"pairs*{args.retrieval}*.txt"))
        pairs_file = candidates[0] if candidates else None
    if pairs_file:
        pairs = read_pairs(pairs_file, query_names, set(map_names))
        # Existing HLoc retrieval pairs preserve order but do not store scores.
        rankings = {query: [(name, None) for name in names[:args.top_k]] for query, names in pairs.items()}
        if any(rankings.values()):
            return rankings
    return rank_from_descriptors(query_global, map_global, args.top_k)


def main() -> None:
    args = parse_args()
    query_out, map_out = args.query / "hloc_outputs", args.map / "hloc_outputs"
    rankings = selected_rankings(args, query_out, map_out)
    query_features = discover_local_features(query_out, args.query_features, "query local features")
    map_features = discover_local_features(map_out, args.map_features, "map local features")
    if args.matches:
        matches_file = args.matches
    else:
        matcher_pattern = "matches*.h5" if args.matcher == "auto" else f"matches*{args.matcher}*.h5"
        # Prefer a match file tagged with the same retrieval method to avoid
        # mixing correspondences generated from another candidate list.
        candidates = sorted(query_out.glob(f"matches*{args.retrieval}*.h5"))
        if args.matcher != "auto":
            candidates = [path for path in candidates if args.matcher.lower() in path.stem.lower()]
        matches_file = candidates[0] if candidates else discover(query_out, matcher_pattern, "local matches")

    selected = list(rankings.items())
    if args.max_queries > 0:
        selected = selected[:args.max_queries]
    local_count = 0
    for query_name, ranked in selected:
        ranked = ranked[:args.top_k]
        if not ranked:
            continue
        safe_query = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(query_name).stem)
        render_contact_sheet(resolve_image(args.query, query_name), ranked, args.map,
                            args.output / "retrieval" / f"{safe_query}.jpg", query_name)
        for map_name, _score in ranked:
            if args.max_local_pairs > 0 and local_count >= args.max_local_pairs:
                break
            safe_map = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(map_name).stem)
            output = args.output / "matches" / f"{safe_query}__{safe_map}.jpg"
            if render_matches(args.query, args.map, query_name, map_name, matches_file,
                              query_features, map_features, output, args.ransac_threshold):
                local_count += 1
    print(f"Saved retrieval images for {len(selected)} queries and {local_count} match images to {args.output}")


if __name__ == "__main__":
    main()

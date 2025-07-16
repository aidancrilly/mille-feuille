#include <mpi.h>
#include "nlohmann/json.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Forrester test function (scalar variant)
// ---------------------------------------------------------------------------
inline double forrester(double x) {
    return std::pow(6.0 * x - 2.0, 2) * std::sin(12.0 * x + 4.0);
}

int main(int argc, char **argv) {
    // ──────────────────────────────────────────────────────────────── MPI init
    MPI_Init(&argc, &argv);

    int nproc = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ───────────────────────────────────────────────────────── Cmd‑line check
    if (rank == 0 && argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.json> <output.txt>\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string fname_in, fname_out;
    if (rank == 0) {
        fname_in  = argv[1];
        fname_out = argv[2];
    }

    // ─────────────────────────────────────────────── Root reads JSON (X, S)
    double X = 0.0; // scalar input
    int    S = 0;

    if (rank == 0) {
        std::ifstream fin(fname_in);
        if (!fin) {
            std::cerr << "Cannot open input file '" << fname_in << "'.\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        json j;
        fin >> j;
        try {
            X = j.at("inputs").at("X").get<double>();
            S = j.at("inputs").at("S").get<int>();
        } catch (const json::exception &e) {
            std::cerr << "Error parsing JSON: " << e.what() << '\n';
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    // ─────────────────────────────────────────────── Broadcast X and S
    MPI_Bcast(&X, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&S, 1, MPI_INT,    0, MPI_COMM_WORLD);

    // ───────────────────────────────────────────── Compute local Y
    const double y0 = forrester(X);

    const double A = 1.0 - (1 - S) * 0.5;
    const double B = (1 - S) * 10.0;
    const double C = (1 - S) * 5.0;

    double Y = -(A * y0 + B * (X - 0.5) + C);

    // ─────────────────────────────────────────── Reduce (sum) then average
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &Y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        Y /= static_cast<double>(nproc); // average across ranks
    } else {
        MPI_Reduce(&Y, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // ─────────────────────────────────────────────────────── Output (root)
    if (rank == 0) {
        std::ofstream fout(fname_out);
        if (!fout) {
            std::cerr << "Cannot open output file '" << fname_out << "'.\n";
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
        fout << Y << '\n';
    }

    MPI_Finalize();
    return 0;
}

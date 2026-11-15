#ifndef FUNCTII_H
#define FUNCTII_H

#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <utility>
#include <iostream>
#include <string>

using namespace std;

inline vector<vector<float>> genereaza_ponderi(int linii, int coloane, mt19937& gen)
{
    vector<vector<float>> W(linii, vector<float>(coloane));
    uniform_real_distribution<float> dis(-0.1f, 0.1f);
    
    for (int i = 0; i < linii; i++)
        for (int j = 0; j < coloane; j++)
            W[i][j] = dis(gen);
    
    return W;
}

inline vector<vector<float>> genereaza_ponderi(int linii, int coloane)
{
    random_device rd;
    mt19937 gen(rd());
    return genereaza_ponderi(linii, coloane, gen);
}

inline vector<vector<float>> genereaza_zerouri(int linii, int coloane)
{
    return vector<vector<float>>(linii, vector<float>(coloane, 0.0f));
}

inline void amesteca_date(vector<vector<float>>& X, vector<vector<float>>& Y)
{
    vector<size_t> indecsi(X.size());
    iota(indecsi.begin(), indecsi.end(), 0);
    random_device rd;
    mt19937 g(rd());
    shuffle(indecsi.begin(), indecsi.end(), g);
    
    vector<vector<float>> X_nou(X.size());
    vector<vector<float>> Y_nou(Y.size());
    
    for (size_t i = 0; i < indecsi.size(); i++)
    {
        X_nou[i] = move(X[indecsi[i]]);
        Y_nou[i] = move(Y[indecsi[i]]);
    }
    
    X = move(X_nou);
    Y = move(Y_nou);
}

inline float sigmoid_scalar(float x)
{
    if (x >= 0.0f)
        return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}

inline vector<vector<float>> sigmoid_mat(const vector<vector<float>>& Z)
{
    vector<vector<float>> H = Z;
    
    for (size_t i = 0; i < H.size(); i++)
        for (size_t j = 0; j < H[0].size(); j++)
            H[i][j] = sigmoid_scalar(H[i][j]);
    
    return H;
}

inline float softplus(float x)
{
    return max(x, 0.0f) + log1p(exp(-abs(x)));
}

inline vector<vector<float>> softmax(const vector<vector<float>>& Z)
{
    vector<vector<float>> P = Z;
    
    for (size_t i = 0; i < P.size(); i++)
    {
        float max_val = P[i][0];
        
        for (size_t j = 1; j < P[0].size(); j++)
            if (P[i][j] > max_val)
                max_val = P[i][j];
            
        float suma_exp = 0.0f;
        
        for (size_t j = 0; j < P[0].size(); j++)
        {
            P[i][j] = exp(P[i][j] - max_val);
            suma_exp += P[i][j];
        }
        
        for (size_t j = 0; j < P[0].size(); j++)
            P[i][j] /= suma_exp;
    }
    
    return P;
}

inline vector<vector<float>> derivata_sigmoid(const vector<vector<float>>& H)
{
    vector<vector<float>> D = H;
    
    for (size_t i = 0; i < H.size(); i++)
        for (size_t j = 0; j < H[0].size(); j++)
            D[i][j] = H[i][j] * (1.0f - H[i][j]);
    
    return D;
}

inline vector<vector<float>> matmul(const vector<vector<float>>& A, const vector<vector<float>>& B)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B[0].size();
    
    vector<vector<float>> C(M, vector<float>(N, 0.0f));
    
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
        {
            float val_A = A[i][k];
            
            for (size_t j = 0; j < N; j++)
                C[i][j] += val_A * B[k][j];
        }
    
    return C;
}

inline vector<vector<float>> transpose(const vector<vector<float>>& A)
{
    vector<vector<float>> T(A[0].size(), vector<float>(A.size()));
    
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            T[j][i] = A[i][j];
    
    return T;
}

inline vector<vector<float>> aduna_bias(const vector<vector<float>>& Z, const vector<vector<float>>& b)
{
    vector<vector<float>> R = Z;
    
    for (size_t i = 0; i < R.size(); i++)
        for (size_t j = 0; j < R[0].size(); j++)
            R[i][j] += b[0][j];
    
    return R;
}

inline vector<vector<float>> suma_liniilor(const vector<vector<float>>& D)
{
    vector<vector<float>> b(1, vector<float>(D[0].size(), 0.0f));
    
    for (size_t i = 0; i < D.size(); i++)
        for (size_t j = 0; j < D[0].size(); j++)
            b[0][j] += D[i][j];
    
    return b;
}

inline vector<vector<float>> mat_sub(const vector<vector<float>>& A, const vector<vector<float>>& B)
{
    vector<vector<float>> C = A;
    
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            C[i][j] -= B[i][j];
    
    return C;
}

inline vector<vector<float>> mat_add(const vector<vector<float>>& A, const vector<vector<float>>& B)
{
    vector<vector<float>> C = A;
    
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            C[i][j] += B[i][j];
    
    return C;
}

inline vector<vector<float>> mat_mul_elem(const vector<vector<float>>& A, const vector<vector<float>>& B)
{
    vector<vector<float>> C = A;
    
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            C[i][j] *= B[i][j];
    
    return C;
}

inline vector<vector<float>> mat_scalar_mul(const vector<vector<float>>& A, float scalar)
{
    vector<vector<float>> C = A;
    
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            C[i][j] *= scalar;
    
    return C;
}

inline vector<int> argmax_linii(const vector<vector<float>>& P)
{
    vector<int> predictii(P.size());
    
    for (size_t i = 0; i < P.size(); i++)
    {
        int max_idx = 0;
        float max_val = P[i][0];
        
        for (size_t j = 1; j < P[0].size(); j++)
            if (P[i][j] > max_val)
            {
                max_val = P[i][j];
                max_idx = j;
            }
        
        predictii[i] = max_idx;
    }
    
    return predictii;
}

inline float cross_entropy(const vector<vector<float>>& P, const vector<vector<float>>& Y)
{
    float pierdere = 0.0f;
    
    for (size_t i = 0; i < P.size(); i++)
        for (size_t j = 0; j < P[0].size(); j++)
            if (Y[i][j] == 1.0f)
                pierdere -= log(P[i][j] + 1e-7f); 
    
    return pierdere / P.size();
}

using Matrice = vector<vector<float>>;

inline double produs_scalar(const Matrice& A, const Matrice& B)
{
    double suma = 0.0;
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[i].size(); j++)
            suma += static_cast<double>(A[i][j]) * B[i][j];
    return suma;
}

inline double pierdere_logits(const Matrice& Z, const Matrice& Y)
{
    double J = 0.0;
    for (size_t i = 0; i < Z.size(); i++)
    {
        double maxim = Z[i][0];
        for (size_t j = 1; j < Z[i].size(); j++)
            if (Z[i][j] > maxim)
                maxim = Z[i][j];

        double suma = 0.0;
        for (size_t j = 0; j < Z[i].size(); j++)
            suma += exp(static_cast<double>(Z[i][j]) - maxim);

        double log_suma = maxim + log(suma);
        for (size_t j = 0; j < Z[i].size(); j++)
            J += Y[i][j] * (log_suma - Z[i][j]);
    }

    J = J / Z.size();
    return J;
}

inline void afiseaza_scor(const string& nume, const Matrice& X, const Matrice& Y,
                         const Matrice& W1, const Matrice& b1,
                         const Matrice& W2, const Matrice& b2)
{
    double J = 0.0;
    size_t corecte = 0;
    for (size_t inceput = 0; inceput < X.size(); inceput += 256)
    {
        size_t sfarsit = min(inceput + 256, X.size());
        Matrice X_batch(X.begin() + inceput, X.begin() + sfarsit);
        Matrice Y_batch(Y.begin() + inceput, Y.begin() + sfarsit);

        Matrice Z1 = matmul(X_batch, W1);
        Z1 = aduna_bias(Z1, b1);
        Matrice H1 = sigmoid_mat(Z1);
        Matrice Z2 = matmul(H1, W2);
        Z2 = aduna_bias(Z2, b2);
        Matrice P = softmax(Z2);

        J += pierdere_logits(Z2, Y_batch) * static_cast<double>(X_batch.size());
        vector<int> prezis = argmax_linii(P);
        vector<int> adevarat = argmax_linii(Y_batch);
        for (size_t i = 0; i < prezis.size(); i++)
            if (prezis[i] == adevarat[i])
                corecte++;
    }
    J = J / X.size();
    double acuratete = 100.0 * static_cast<double>(corecte) / X.size();
    cout << " J_" << nume << "=" << J;
    cout << " acc_" << nume << "=" << acuratete << "%";
}

#endif

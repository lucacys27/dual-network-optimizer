#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "Functii.h"

using namespace std;

bool citeste_uint32(istream &fisier, uint32_t &val)
{
    unsigned char octeti[4];
    if (!fisier.read(reinterpret_cast<char *>(octeti), sizeof(octeti)))
        return false;

    val = (static_cast<uint32_t>(octeti[0]) << 24) |
          (static_cast<uint32_t>(octeti[1]) << 16) |
          (static_cast<uint32_t>(octeti[2]) << 8) |
          static_cast<uint32_t>(octeti[3]);
    return true;
}

bool verifica_lungime(ifstream &fisier, uint64_t octeti_asteptati)
{
    const auto inceput = fisier.tellg();
    fisier.seekg(0, ios::end);
    const auto sfarsit = fisier.tellg();
    if (inceput == streampos(-1) || sfarsit < inceput ||
        static_cast<uint64_t>(sfarsit - inceput) != octeti_asteptati)
        return false;

    return static_cast<bool>(fisier.seekg(inceput));
}

bool citeste_imagini(ifstream &fisier, Matrice &X)
{
    uint32_t magic, numar, linii, coloane;
    if (!citeste_uint32(fisier, magic) || !citeste_uint32(fisier, numar) ||
        !citeste_uint32(fisier, linii) || !citeste_uint32(fisier, coloane) ||
        magic != 2051 || numar == 0 || linii != 28 || coloane != 28)
        return false;

    unsigned char pixeli[28 * 28];
    if (!verifica_lungime(fisier, static_cast<uint64_t>(numar) * sizeof(pixeli)))
        return false;

    X.resize(numar, vector<float>(sizeof(pixeli)));
    for (auto &imagine : X)
    {
        if (!fisier.read(reinterpret_cast<char *>(pixeli), sizeof(pixeli)))
            return false;

        for (size_t j = 0; j < imagine.size(); j++)
            imagine[j] = static_cast<float>(pixeli[j]) / 255.0f;
    }
    return true;
}

bool citeste_etichete(ifstream &fisier, size_t numar_imagini, vector<int> &Y)
{
    uint32_t magic, numar;
    if (!citeste_uint32(fisier, magic) || !citeste_uint32(fisier, numar) ||
        magic != 2049 || numar != numar_imagini || !verifica_lungime(fisier, numar))
        return false;

    Y.resize(numar);
    for (int &eticheta : Y)
    {
        int cifra = fisier.get();
        if (cifra < 0 || cifra > 9)
            return false;
        eticheta = cifra;
    }
    return true;
}

int main()
{
    const char *fisiere[] = {
        "train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"
    };
    for (const char *nume : fisiere)
    {
        if (!ifstream(nume, ios::binary))
        {
            cerr << "Nu pot deschide " << nume << ". Vezi instructiunile MNIST din README.\n";
            return 1;
        }
    }

    ifstream file("train-images.idx3-ubyte", ios::binary);
    Matrice X_train;
    if (!citeste_imagini(file, X_train))
    {
        cerr << "Fisier IDX invalid: train-images.idx3-ubyte\n";
        return 1;
    }
    file.close();

    ifstream label_file("train-labels.idx1-ubyte", ios::binary);
    vector<int> etichete_train;
    if (!citeste_etichete(label_file, X_train.size(), etichete_train))
    {
        cerr << "Fisier IDX invalid: train-labels.idx1-ubyte\n";
        return 1;
    }
    label_file.close();

    Matrice Y_train(X_train.size(), vector<float>(10, 0.0f));
    for (size_t i = 0; i < X_train.size(); i++)
        Y_train[i][etichete_train[i]] = 1.0f;

    int epoci = 5;
    int marime_batch = 64;
    float alpha = 0.001f;

    int intrari = X_train[0].size();
    int clase = Y_train[0].size();
    int ascunsi = 64;
    mt19937 rng_model(123);

    Matrice W1 = genereaza_ponderi(intrari, ascunsi, rng_model);
    Matrice b1 = genereaza_zerouri(1, ascunsi);
    Matrice W2 = genereaza_ponderi(ascunsi, clase, rng_model);
    Matrice b2 = genereaza_zerouri(1, clase);

    Matrice Wt1 = genereaza_ponderi(5, 8, rng_model);
    Matrice bt1 = genereaza_zerouri(1, 8);
    Matrice Wt2 = genereaza_ponderi(8, 2, rng_model);
    Matrice bt2 = genereaza_zerouri(1, 2);
    Wt2 = mat_scalar_mul(Wt2, 0.1f);
    bt2[0][0] = log(expm1(0.01f));
    bt2[0][1] = log(0.9f / 0.1f);

    Matrice vW1 = genereaza_zerouri(intrari, ascunsi);
    Matrice vb1 = genereaza_zerouri(1, ascunsi);
    Matrice vW2 = genereaza_zerouri(ascunsi, clase);
    Matrice vb2 = genereaza_zerouri(1, clase);
    float eta_precedent = 0.01f;
    float beta_precedent = 0.9f;

    vector<size_t> ordine(X_train.size());
    iota(ordine.begin(), ordine.end(), 0);
    mt19937 rng_amestecare(456);
    int pasi_epoca = (X_train.size() + marime_batch - 1) / marime_batch;
    int total_pasi = epoci * pasi_epoca;
    int it = 0;
    vector<double> vJ;
    vector<float> vEta;
    vector<float> vBeta;

    for (int epoca = 0; epoca < epoci; epoca++)
    {
        shuffle(ordine.begin(), ordine.end(), rng_amestecare);

        for (size_t inceput = 0; inceput < X_train.size(); inceput += marime_batch)
        {
            size_t sfarsit = min(inceput + marime_batch, X_train.size());
            Matrice X;
            Matrice Y;
            for (size_t i = inceput; i < sfarsit; i++)
            {
                size_t k = ordine[i];
                X.push_back(X_train[k]);
                Y.push_back(Y_train[k]);
            }
            int N = X.size();

            Matrice Z1 = matmul(X, W1);
            Z1 = aduna_bias(Z1, b1);
            Matrice H1 = sigmoid_mat(Z1);

            Matrice Z2 = matmul(H1, W2);
            Z2 = aduna_bias(Z2, b2);
            Matrice P = softmax(Z2);
            double J = pierdere_logits(Z2, Y);

            Matrice gZ2 = mat_sub(P, Y);
            gZ2 = mat_scalar_mul(gZ2, 1.0f / N);

            Matrice H1_transpus = transpose(H1);
            Matrice gW2 = matmul(H1_transpus, gZ2);
            Matrice gb2 = suma_liniilor(gZ2);

            Matrice W2_transpus = transpose(W2);
            Matrice gH1 = matmul(gZ2, W2_transpus);
            Matrice derivata_H1 = derivata_sigmoid(H1);
            Matrice gZ1 = mat_mul_elem(gH1, derivata_H1);

            Matrice X_transpus = transpose(X);
            Matrice gW1 = matmul(X_transpus, gZ1);
            Matrice gb1 = suma_liniilor(gZ1);

            Matrice S = genereaza_zerouri(1, 5);

            double norma2_g = produs_scalar(gW1, gW1);
            norma2_g += produs_scalar(gb1, gb1);
            norma2_g += produs_scalar(gW2, gW2);
            norma2_g += produs_scalar(gb2, gb2);
            double norma_g = sqrt(norma2_g);

            float progres = float(it) / max(1, total_pasi - 1);

            S[0][0] = log1p(J);
            S[0][1] = log1p(norma_g);
            S[0][2] = eta_precedent;
            S[0][3] = beta_precedent;
            S[0][4] = progres;

            Matrice Zt1 = matmul(S, Wt1);
            Zt1 = aduna_bias(Zt1, bt1);
            Matrice Ht1 = sigmoid_mat(Zt1);

            Matrice Zt2 = matmul(Ht1, Wt2);
            Zt2 = aduna_bias(Zt2, bt2);
            float eta = softplus(Zt2[0][0]);
            float beta = sigmoid_scalar(Zt2[0][1]);

            Matrice vW1_vechi = vW1;
            Matrice vb1_vechi = vb1;
            Matrice vW2_vechi = vW2;
            Matrice vb2_vechi = vb2;

            vW1 = mat_scalar_mul(vW1, beta);
            vW1 = mat_add(vW1, gW1);
            vb1 = mat_scalar_mul(vb1, beta);
            vb1 = mat_add(vb1, gb1);
            vW2 = mat_scalar_mul(vW2, beta);
            vW2 = mat_add(vW2, gW2);
            vb2 = mat_scalar_mul(vb2, beta);
            vb2 = mat_add(vb2, gb2);

            Matrice pas_W1 = mat_scalar_mul(vW1, eta);
            Matrice pas_b1 = mat_scalar_mul(vb1, eta);
            Matrice pas_W2 = mat_scalar_mul(vW2, eta);
            Matrice pas_b2 = mat_scalar_mul(vb2, eta);
            W1 = mat_sub(W1, pas_W1);
            b1 = mat_sub(b1, pas_b1);
            W2 = mat_sub(W2, pas_W2);
            b2 = mat_sub(b2, pas_b2);

            Z1 = matmul(X, W1);
            Z1 = aduna_bias(Z1, b1);
            H1 = sigmoid_mat(Z1);

            Z2 = matmul(H1, W2);
            Z2 = aduna_bias(Z2, b2);
            P = softmax(Z2);
            double J_dupa = pierdere_logits(Z2, Y);

            gZ2 = mat_sub(P, Y);
            gZ2 = mat_scalar_mul(gZ2, 1.0f / N);

            H1_transpus = transpose(H1);
            Matrice gW2_dupa = matmul(H1_transpus, gZ2);
            Matrice gb2_dupa = suma_liniilor(gZ2);

            W2_transpus = transpose(W2);
            gH1 = matmul(gZ2, W2_transpus);
            derivata_H1 = derivata_sigmoid(H1);
            gZ1 = mat_mul_elem(gH1, derivata_H1);

            Matrice gW1_dupa = matmul(X_transpus, gZ1);
            Matrice gb1_dupa = suma_liniilor(gZ1);

            double q_eta = produs_scalar(gW1_dupa, vW1);
            q_eta += produs_scalar(gb1_dupa, vb1);
            q_eta += produs_scalar(gW2_dupa, vW2);
            q_eta += produs_scalar(gb2_dupa, vb2);
            q_eta = -q_eta;

            double q_beta = produs_scalar(gW1_dupa, vW1_vechi);
            q_beta += produs_scalar(gb1_dupa, vb1_vechi);
            q_beta += produs_scalar(gW2_dupa, vW2_vechi);
            q_beta += produs_scalar(gb2_dupa, vb2_vechi);
            q_beta = -eta * q_beta;

            Matrice gZt2 = genereaza_zerouri(1, 2);
            gZt2[0][0] = q_eta * sigmoid_scalar(Zt2[0][0]);
            gZt2[0][1] = q_beta * beta * (1.0f - beta);

            Matrice Ht1_transpus = transpose(Ht1);
            Matrice gWt2 = matmul(Ht1_transpus, gZt2);
            Matrice gbt2 = gZt2;

            Matrice Wt2_transpus = transpose(Wt2);
            Matrice gHt1 = matmul(gZt2, Wt2_transpus);
            Matrice derivata_Ht1 = derivata_sigmoid(Ht1);
            Matrice gZt1 = mat_mul_elem(gHt1, derivata_Ht1);

            Matrice S_transpus = transpose(S);
            Matrice gWt1 = matmul(S_transpus, gZt1);
            Matrice gbt1 = gZt1;

            Matrice pas_Wt1 = mat_scalar_mul(gWt1, alpha);
            Matrice pas_bt1 = mat_scalar_mul(gbt1, alpha);
            Matrice pas_Wt2 = mat_scalar_mul(gWt2, alpha);
            Matrice pas_bt2 = mat_scalar_mul(gbt2, alpha);
            Wt1 = mat_sub(Wt1, pas_Wt1);
            bt1 = mat_sub(bt1, pas_bt1);
            Wt2 = mat_sub(Wt2, pas_Wt2);
            bt2 = mat_sub(bt2, pas_bt2);

            eta_precedent = eta;
            beta_precedent = beta;
            vJ.push_back(J_dupa);
            vEta.push_back(eta);
            vBeta.push_back(beta);
            it++;
        }

        cout << "Epoca " << epoca + 1 << ":";
        afiseaza_scor("train", X_train, Y_train, W1, b1, W2, b2);
        cout << " eta=" << eta_precedent;
        cout << " beta=" << beta_precedent << '\n';
    }

    ifstream test_file("t10k-images.idx3-ubyte", ios::binary);
    Matrice X_test;
    if (!citeste_imagini(test_file, X_test))
    {
        cerr << "Fisier IDX invalid: t10k-images.idx3-ubyte\n";
        return 1;
    }
    test_file.close();

    ifstream test_label_file("t10k-labels.idx1-ubyte", ios::binary);
    vector<int> Y_test;
    if (!citeste_etichete(test_label_file, X_test.size(), Y_test))
    {
        cerr << "Fisier IDX invalid: t10k-labels.idx1-ubyte\n";
        return 1;
    }
    test_label_file.close();

    size_t num_images_test = X_test.size();
    Matrice Z1_test = matmul(X_test, W1);
    Z1_test = aduna_bias(Z1_test, b1);
    Matrice H1_test = sigmoid_mat(Z1_test);

    Matrice Z2_test = matmul(H1_test, W2);
    Z2_test = aduna_bias(Z2_test, b2);
    Matrice P_test = softmax(Z2_test);

    vector<int> predictii = argmax_linii(P_test);

    size_t corecte = 0;
    for (size_t i = 0; i < num_images_test; i++)
    {
        if (predictii[i] == Y_test[i])
            corecte++;
    }

    double acuratete_test = 100.0 * corecte / num_images_test;
    cout << "Corecte pe test: " << corecte << " / " << num_images_test << '\n';
    cout << "Acuratete test: " << acuratete_test << "%\n";

    return 0;
}

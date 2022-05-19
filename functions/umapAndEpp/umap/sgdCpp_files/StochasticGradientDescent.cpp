/*
 *
   AUTHORSHIP
   Primary Developers: 	 Connor Meehan <connor.gw.meehan@gmail.com> 
   			 Stephen Meehan <swmeehan@stanford.edu> 
   Math Lead:  		 Connor Meehan <connor.gw.meehan@gmail.com> 
   Bioinformatics Lead:  Wayne Moore <wmoore@stanford.edu>
   Provided by the Herzenberg Lab at Stanford University 
   License: BSD 3 clause
   
   cm.gautham@yahoo.in did the first C++ translation from the prior
   StochasticGradientDescent.java.  The core java translated 
   almost line for line into C++.  Thus Gautham's translation
   became the seed that with much additional work lead to a first
   working implementation complete with 2D and 3D output.

*/

#include <iostream>
#include <fstream>
#include <math.h>
#include<stdio.h>
#include<string.h>
#include <exception>

template <typename T>
T** create2DArray(unsigned nrows, unsigned ncols, const T& val = T())
{
    if (nrows == 0)
        throw std::invalid_argument("number of rows is 0");
    if (ncols == 0)
        throw std::invalid_argument("number of columns is 0");
    T** ptr = nullptr;
    T* pool = nullptr;
    try
    {
        ptr = new T*[nrows];  // allocate pointers (can throw here)
        pool = new T[nrows*ncols];  // allocate pool (can throw here)

        // now point the row pointers to the appropriate positions in
        // the memory pool
        for (unsigned i = 0; i < nrows; ++i, pool += ncols )
            ptr[i] = pool;

        // Done.
        return ptr;
    }
    catch (std::bad_alloc& ex)
    {
        delete [] ptr; // either this is nullptr or it was allocated
        throw ex;  // memory allocation error
    }
}

template <typename T>
void delete2DArray(T** arr)
{
    delete [] arr[0];  // remove the pool
    delete [] arr;     // remove the pointers
}

char *outputFile=NULL;
int rows, cols;
bool stopped = false;

int checkForStop(){
    FILE *stop;
    char *stopn=NULL;
    stopn = (char *) malloc(strlen(outputFile) + 25);
    sprintf(stopn, "%s.%s", outputFile, "STOP");
    if ((stop = fopen(stopn, "r+b")) != NULL) {
        stopped = true;
        fclose(stop);
        remove(stopn);
    }
    if (stopped) return -2;
    return 0;
}

int save(double **matrix, int epochs){
    checkForStop();
    if (stopped) return -2;

    FILE *out;
    char *fn=NULL;

    if (epochs>0) {
        fn = (char *) malloc(strlen(outputFile) + 25);
        sprintf(fn, "%s.%d", outputFile, epochs);
    } else{
        fn=outputFile;
    }
    if ((out = fopen(fn, "w+b")) == NULL) {
        printf("Error: could not open data file.\n");
        return -1;
    }

    fprintf(out, "%d\n", rows);
    fprintf(out, "%d\n", cols);

    //Write all elements of each dimension to support reading using MatLab's reshape method
    for (register int col = 0; col < cols; col++) {
        for (register int row = 0; row < rows; row++) {
            fprintf(out, "%lf\n", matrix[row][col]);
        }
    }
    fclose(out);
    return 0;
}

/* START debug section for comparing with Java*/
bool DEBUGGING = false;
int verbose=0;
int const MAX_VERBOSE=5;

// set the number of debug prints
int debugPrintLimit=15;
int save_progress=0;


// set which events/indexes in head_embedding to do debug 
// prints for.... this assumes main is looking at
//CytoGate\src\edu\stanford\facs\swing\sgdInput.txt
// This file is kept in CVS>
//	It was generated from sample2k.csv
//... in MatLab we produced sgdInput.txt by running
//	run_umap('sample2k.csv', 'method', 'C++');
//	the sample2k.csv is in CVS folder CytoGate/matlabsrc/umap
int debugWatch[]={0, 1999};
// must set debugWatchLength since C++ does not have debugWatch.length
int debugWatchLength=2;

bool IsDebugIdx(int idx) {
    for (int i=0;i<debugWatchLength;i++) {
        if (debugWatch[i]==idx) {
            return true;
        }
    }
    return false;
}

bool doDebug = false;
int debugPrints = 0;

void debug2D(char *title,  double **a, const int n) {
    printf("%s 1st/end [%lf %lf]/[%lf %lf]\n", title,
           a[0][0],
           a[0][1],
           a[n - 1][0],
           a[n - 1][1]);
}

void debug1D(char *title, int *v, int n) {
    printf("%s 1st/end %d/%d\n", title, v[0], v[n-1]);
}

void debug1D(char *title, double *v, int n) {
    printf("%s 1st/end %lf/%lf\n", title, v[0], v[n-1]);
}

void debugHeadTail(const bool move_other,  double **head_embedding, const int h,  double **tail_embedding, const int t) {
    debug2D("head", head_embedding, h);
    printf("\t\t");
    if (!move_other){
    	debug2D("tail", tail_embedding, t);
    }
}

/* END debug section for comparing with Java*/
void doVerbose(
        const int verbose, const bool move_other,
        const int size_head_embedding, const int size_tail_embedding,
         double **head_embedding,  double **tail_embedding,
        const int *head, const int *tail, const int n_epochs, const int n_vertices,
        const double *epochs_per_sample, const double a, const double b,
        const double gamma, const double initial_alpha,
        const int negative_sample_rate, const int n_1_simplices,
        const int n_components) {
    //RUN LOGIC
    if (verbose>0)
        debugHeadTail(move_other, head_embedding, size_head_embedding, tail_embedding, size_tail_embedding);
    double alpha;
    double BG2S;
    double ABNEG2;
    double BNEG1;
    double *epochs_per_negative_sample;
    double *epoch_of_next_negative_sample;
    double *epoch_of_next_sample;
    double nTh;
    int n_epoch;
    const int EPOCH_REPORTS = 10;//20;

    alpha = initial_alpha;
    BG2S = 2 * gamma * b;
    ABNEG2 = -2.0 * a * b;
    BNEG1 = b - 1;
    if (verbose>0)
        printf("Perform logic BG2S=%f, ABNEG2=%f BNEG1=%f.\n", BG2S, ABNEG2, BNEG1);
    epoch_of_next_negative_sample = (double *) malloc(n_1_simplices * sizeof(double));
    epochs_per_negative_sample = (double *) malloc(n_1_simplices * sizeof(double));
    epoch_of_next_sample = (double *) malloc(n_1_simplices * sizeof(double));

    for (int i = 0; i < n_1_simplices; i++) {
        epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
    }
    for (int i = 0; i < n_1_simplices; i++) {
        epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i];
        epoch_of_next_sample[i] = epochs_per_sample[i];
    }
    nTh = (double) n_epochs / double(EPOCH_REPORTS);
    if (verbose>>0)
        printf("nTh value %lf\n", nTh);
    n_epoch = 1;

    bool hasNextEphos = true;

    while (hasNextEphos) {
        int iRandi = 0;
        double current[n_components];
        double other[n_components];
        int n_neg_samples = 0;
        double grad[n_components];
        double sub[n_components];
        double grad_coef = 0;
        double dist_squared = 0;
        double val = 0;
        double alpha4 = alpha * 4, alphaNeg4 = alpha * -4;
        for (register int m = 0; m < n_components; m++) {
            current[m] = 0;
            other[m] = 0;
            grad[m] = 0;
            sub[m] = 0;
        }
        /*if (DEBUGGING) {
            printf("n_epoch, n_epochs %d, %d, %d %d %d\n", n_epoch, n_epochs, n_vertices, n_1_simplices, n_components);
        }*/
        int computes = 0;
        for (int n = n_epoch; n <= n_epochs; n++) {
            for (int i = 0; i < n_1_simplices; i++) {
                if (epoch_of_next_sample[i] > n) {
                    continue;
                }
                const int j = head[i] - 1;//const
                int k = tail[i] - 1;
                for (register int m = 0; m < n_components; m++) {
                    current[m] = head_embedding[j][m];
                    other[m] = tail_embedding[k][m];
                    sub[m] = current[m] - other[m];
                }
                dist_squared = 0;
                for (register int m = 0; m < n_components; m++) {
                    dist_squared += sub[m] * sub[m];
                }
                if (dist_squared > 0) {
                    /*if (DEBUGGING) {
                        //doDebug=j==0 && n>7 && n<10  && debugPrints<15;
                        doDebug =(IsDebugIdx(j) || IsDebugIdx(k)) && n>5 && debugPrints<debugPrintLimit;
                        if (doDebug) {
                            debugPrints++;
                            printf("debugPrint #%d:  n=%d, j=%d, k=%d, sub=[%f %f] current=[%f %f], other=[%f %f], dist_squared=%f \n ",
                                   debugPrints, n, j, k, sub[0], sub[1], current[0], current[1], other[0], other[1], dist_squared);
                        }
                    }*/
                    grad_coef = (ABNEG2 * pow(dist_squared, BNEG1)) / (a * pow(dist_squared, b) + 1);
                    for (register int m = 0; m < n_components; m++) {
                        val = grad_coef * sub[m];
                        if (val >= 4) {
                            grad[m] = alpha4;
                        } else if (val <= -4) {
                            grad[m] = alphaNeg4;
                        } else {
                            grad[m] = val * alpha;
                        }
                        current[m] = current[m] + grad[m];
                    }
                    /*if (DEBUGGING) {
                        if (doDebug) {
                            printf("  ...grad=[%f %f] current=[%f %f] \n ", grad[0], grad[1], current[0], current[1]);
                        }
                    }*/
                    if (move_other) {
                        for (register int m = 0; m < n_components; m++) {
                            other[m] = other[m] - grad[m];
                            tail_embedding[k][m] = other[m];
                        }
                    }
                } else {
                    for (register int m = 0; m < n_components; m++) {
                        grad[m] = 0;
                    }
                }
                epoch_of_next_sample[i] += epochs_per_sample[i];
                n_neg_samples = static_cast<int>(floor(((static_cast<double>(n))
                                                        - epoch_of_next_negative_sample[i]) /
                                                       epochs_per_negative_sample[i]));

                for (int p = 0; p < n_neg_samples; p++) {
                    k = rand() % n_vertices;
                    /*if (DEBUGGING){
                        if (iRandi >= size_randis) {
                            iRandi = 0;
                        }
                        k = randis[iRandi++];
                    }*/
                    if (move_other && j == k) {
                        continue;
                    }
                    dist_squared = 0;
                    for (register int m = 0; m < n_components; m++) {
                        other[m] = tail_embedding[k][m];
                        sub[m] = current[m] - other[m];
                        dist_squared += sub[m] * sub[m];
                    }
                    if (dist_squared > 0) {
                        grad_coef = ((BG2S / (0.001 + dist_squared))) / (a * pow(dist_squared, b) + 1);
                        for (register int m = 0; m < n_components; m++) {
                            val = grad_coef * sub[m];
                            if (val >= 4) {
                                grad[m] = alpha4;
                            } else if (val <= -4) {
                                grad[m] = alphaNeg4;
                            } else {
                                grad[m] = val * alpha;
                            }
                        }
                    } else {
                        for (register int m = 0; m < n_components; m++) {
                            grad[m] = 4;
                        }
                    }
                    for (register int m = 0; m < n_components; m++) {
                        current[m] = current[m] + (grad[m]);
                    }
                    /*if (DEBUGGING) {
                        if (doDebug) {
                            if (verbose>0)
                                printf("\tp=%d, k=%d: dist_squared=%f, grad=[%f %f], \n\t\tcurrent=[%f %f]\n", p, k,
                                   dist_squared,
                                   grad[0],
                                   grad[1],
                                   current[0],
                                   current[1]);
                        }
                    }*/
                }
                for (register int m = 0; m < n_components; m++) {
                    head_embedding[j][m] = current[m];
                }
                epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
                /*if (DEBUGGING) {
                    computes++;
                }*/
            }
            alpha = initial_alpha * (1 - static_cast<double>(static_cast<double>(n) / static_cast<double>(n_epochs)));
            alpha4 = alpha * 4;
            alphaNeg4 = alpha * -4;
            double nBynTh = floor(fmod((double) n, nTh));
            if (nBynTh == 0) {
                n_epoch = n + 1;
                if (n_epoch < n_epochs) {
                    /*if (DEBUGGING) {
                        printf("%d/%d epochs:\t", (n_epoch - 1), n_epochs);
                        debugHeadTail(move_other, head_embedding, size_head_embedding, tail_embedding, size_tail_embedding);
                        exit(1);
                    } else {*/
                    if (save_progress>0){
                        save(head_embedding, n_epoch);
                    }
                    printf("%d/%d epochs done\n", (n_epoch - 1), n_epochs);
                    if (verbose>0)
                        debugHeadTail(move_other, head_embedding, size_head_embedding, tail_embedding, size_tail_embedding);
                    //}
                    hasNextEphos = false;
                }
            }
            checkForStop();
            if (stopped) break;
        }
        if (stopped) break;
    }

    //SAVE
    if (verbose>0) {
        if (stopped) printf("Stop command received. Halting algorithm...\n");
        else printf("Saving result.\n");
    }
    free(epoch_of_next_negative_sample);
    free(epochs_per_negative_sample);
    free(epoch_of_next_sample);

}

void doMoveOther(
         double **head_embedding,  double **tail_embedding,
        const int *head, const int *tail, int n_epochs, int n_vertices,
        const double *epochs_per_sample, const double a, const double b,
        const double gamma, const double initial_alpha,
        const int negative_sample_rate, const int n_1_simplices,
        const int n_components) {

    //RUN LOGIC
    double alpha;
    double BG2S;
    double ABNEG2;
    double BNEG1;
    double *epochs_per_negative_sample;
    double *epoch_of_next_negative_sample;
    double *epoch_of_next_sample;
    double nTh;
    int n_epoch;
    const int EPOCH_REPORTS = 10;//20;

    alpha = initial_alpha;
    BG2S = 2 * gamma * b;
    ABNEG2 = -2.0 * a * b;
    BNEG1 = b - 1;
    epoch_of_next_negative_sample = (double *) malloc(n_1_simplices * sizeof(double));
    epochs_per_negative_sample = (double *) malloc(n_1_simplices * sizeof(double));
    epoch_of_next_sample = (double *) malloc(n_1_simplices * sizeof(double));

    for (int i = 0; i < n_1_simplices; i++) {
        epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
    }
    for (int i = 0; i < n_1_simplices; i++) {
        epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i];
        epoch_of_next_sample[i] = epochs_per_sample[i];
    }
    nTh = (double) n_epochs / double(EPOCH_REPORTS);
    n_epoch = 1;

    bool hasNextEphos = true;

    while (hasNextEphos) {
        int iRandi = 0;
        double current[n_components];
        double other[n_components];
        int n_neg_samples = 0;
        double grad[n_components];
        double sub[n_components];
        double grad_coef = 0;
        double dist_squared = 0;
        double val = 0;
        double alpha4 = alpha * 4, alphaNeg4 = alpha * -4;
        for (register int m = 0; m < n_components; m++) {
            current[m] = 0;
            other[m] = 0;
            grad[m] = 0;
            sub[m] = 0;
        }
        for (
                int n = n_epoch;
                n <=
                n_epochs;
                n++) {
            for (
                    int i = 0;
                    i < n_1_simplices;
                    i++) {
                if (epoch_of_next_sample[i] > n) {
                    continue;
                }
                const int j = head[i] - 1;//const
                int k = tail[i] - 1;
                for (register int m = 0; m < n_components; m++) {
                    current[m] = head_embedding[j][m];
                    other[m] = tail_embedding[k][m];
                    sub[m] = current[m] - other[m];
                }
                dist_squared = 0;
                for (register int m = 0; m < n_components; m++) {
                    dist_squared += sub[m] * sub[m];
                }
                if (dist_squared > 0) {
                    grad_coef = (ABNEG2 * pow(dist_squared, BNEG1)) / (a * pow(dist_squared, b) + 1);
                    for (register int m = 0; m < n_components; m++) {
                        val = grad_coef * sub[m];
                        if (val >= 4) {
                            grad[m] = alpha4;
                        } else if (val <= -4) {
                            grad[m] = alphaNeg4;
                        } else {
                            grad[m] = val * alpha;
                        }
                        current[m] = current[m] + grad[m];
                    }
                    for (register int m = 0; m < n_components; m++) {
                        other[m] = other[m] - grad[m];
                        tail_embedding[k][m] = other[m];
                    }
                } else {
                    for (register int m = 0; m < n_components; m++) {
                        grad[m] = 0;
                    }
                }
                epoch_of_next_sample[i] += epochs_per_sample[i];
                n_neg_samples = static_cast<int>(floor(((static_cast<double>(n))
                                                        - epoch_of_next_negative_sample[i]) /
                                                       epochs_per_negative_sample[i]));

                for (int p = 0; p < n_neg_samples; p++) {
                    k = rand() % n_vertices;
                    if (j == k) {
                        continue;
                    }
                    dist_squared = 0;
                    for (register int m = 0; m < n_components; m++) {
                        other[m] = tail_embedding[k][m];
                        sub[m] = current[m] - other[m];
                        dist_squared += sub[m] * sub[m];
                    }
                    if (dist_squared > 0) {
                        grad_coef = ((BG2S / (0.001 + dist_squared))) / (a * pow(dist_squared, b) + 1);
                        for (register int m = 0; m < n_components; m++) {
                            val = grad_coef * sub[m];
                            if (val >= 4) {
                                grad[m] = alpha4;
                            } else if (val <= -4) {
                                grad[m] = alphaNeg4;
                            } else {
                                grad[m] = val * alpha;
                            }
                        }
                    } else {
                        for (register int m = 0; m < n_components; m++) {
                            grad[m] = 4;
                        }
                    }
                    for (register int m = 0; m < n_components; m++) {
                        current[m] = current[m] + (grad[m]);
                    }
                }
                for (register int m = 0; m < n_components; m++) {
                    head_embedding[j][m] = current[m];
                }
                epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
            }
            alpha = initial_alpha * (1 - static_cast<double>(static_cast<double>(n) / static_cast<double>(n_epochs)));
            alpha4 = alpha * 4;
            alphaNeg4 = alpha * -4;
            double nBynTh = floor(fmod((double) n, nTh));
            if (nBynTh == 0) {
                n_epoch = n + 1;
                if (n_epoch < n_epochs) {
                    if (save_progress>0){
                        save(head_embedding, n_epoch);
                    }
                    printf("%d/%d epochs done\n", (n_epoch - 1), n_epochs);
                    hasNextEphos = false;
                }
            }
            checkForStop();
            if (stopped) break;
        }
        if (stopped) break;
    }
    free(epoch_of_next_negative_sample);
    free(epochs_per_negative_sample);
    free(epoch_of_next_sample);
}

void doNotMoveOther(
        double **head_embedding,  double **tail_embedding,
        const int *head, const int *tail, int n_epochs, int n_vertices,
        const double *epochs_per_sample, const double a, const double b,
        const double gamma, const double initial_alpha,
        const int negative_sample_rate, const int n_1_simplices,
        const int n_components) {

    //RUN LOGIC
    double alpha;
    double BG2S;
    double ABNEG2;
    double BNEG1;
    double *epochs_per_negative_sample;
    double *epoch_of_next_negative_sample;
    double *epoch_of_next_sample;
    double nTh;
    int n_epoch;
    const int EPOCH_REPORTS = 10;//20;

    alpha = initial_alpha;
    BG2S = 2 * gamma * b;
    ABNEG2 = -2.0 * a * b;
    BNEG1 = b - 1;
    epoch_of_next_negative_sample = (double *) malloc(n_1_simplices * sizeof(double));
    epochs_per_negative_sample = (double *) malloc(n_1_simplices * sizeof(double));
    epoch_of_next_sample = (double *) malloc(n_1_simplices * sizeof(double));

    for (int i = 0; i < n_1_simplices; i++) {
        epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
    }
    for (int i = 0; i < n_1_simplices; i++) {
        epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i];
        epoch_of_next_sample[i] = epochs_per_sample[i];
    }
    nTh = (double) n_epochs / double(EPOCH_REPORTS);
    n_epoch = 1;
    bool hasNextEphos = true;
    while (hasNextEphos) {
        int iRandi = 0;
        double current[n_components];
        double other[n_components];
        int n_neg_samples = 0;
        double grad[n_components];
        double sub[n_components];
        double grad_coef = 0;
        double dist_squared = 0;
        double val = 0;
        double alpha4 = alpha * 4, alphaNeg4 = alpha * -4;
        for (register int m = 0; m < n_components; m++) {
            current[m] = 0;
            other[m] = 0;
            grad[m] = 0;
            sub[m] = 0;
        }
        for (
                int n = n_epoch;
                n <=
                n_epochs;
                n++) {
            for (
                    int i = 0;
                    i < n_1_simplices;
                    i++) {
                if (epoch_of_next_sample[i] > n) {
                    continue;
                }
                const int j = head[i] - 1;//const
                int k = tail[i] - 1;
                for (register int m = 0; m < n_components; m++) {
                    current[m] = head_embedding[j][m];
                    other[m] = tail_embedding[k][m];
                    sub[m] = current[m] - other[m];
                }
                dist_squared = 0;
                for (register int m = 0; m < n_components; m++) {
                    dist_squared += sub[m] * sub[m];
                }
                if (dist_squared > 0) {
                    grad_coef = (ABNEG2 * pow(dist_squared, BNEG1)) / (a * pow(dist_squared, b) + 1);
                    for (register int m = 0; m < n_components; m++) {
                        val = grad_coef * sub[m];
                        if (val >= 4) {
                            grad[m] = alpha4;
                        } else if (val <= -4) {
                            grad[m] = alphaNeg4;
                        } else {
                            grad[m] = val * alpha;
                        }
                        current[m] = current[m] + grad[m];
                    }
                } else {
                    for (register int m = 0; m < n_components; m++) {
                        grad[m] = 0;
                    }
                }
                epoch_of_next_sample[i] += epochs_per_sample[i];
                n_neg_samples = static_cast<int>(floor(((static_cast<double>(n))
                                                        - epoch_of_next_negative_sample[i]) /
                                                       epochs_per_negative_sample[i]));

                for (int p = 0; p < n_neg_samples; p++) {
                    k = rand() % n_vertices;
                    dist_squared = 0;
                    for (register int m = 0; m < n_components; m++) {
                        other[m] = tail_embedding[k][m];
                        sub[m] = current[m] - other[m];
                        dist_squared += sub[m] * sub[m];
                    }
                    if (dist_squared > 0) {
                        grad_coef = ((BG2S / (0.001 + dist_squared))) / (a * pow(dist_squared, b) + 1);
                        for (register int m = 0; m < n_components; m++) {
                            val = grad_coef * sub[m];
                            if (val >= 4) {
                                grad[m] = alpha4;
                            } else if (val <= -4) {
                                grad[m] = alphaNeg4;
                            } else {
                                grad[m] = val * alpha;
                            }
                        }
                    } else {
                        for (register int m = 0; m < n_components; m++) {
                            grad[m] = 4;
                        }
                    }
                    for (register int m = 0; m < n_components; m++) {
                        current[m] = current[m] + (grad[m]);
                    }
                }
                for (register int m = 0; m < n_components; m++) {
                    head_embedding[j][m] = current[m];
                }
                epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
            }
            alpha = initial_alpha * (1 - static_cast<double>(static_cast<double>(n) / static_cast<double>(n_epochs)));
            alpha4 = alpha * 4;
            alphaNeg4 = alpha * -4;
            double nBynTh = floor(fmod((double) n, nTh));
            if (nBynTh == 0) {
                n_epoch = n + 1;
                if (n_epoch < n_epochs) {
                    if (save_progress>0){
                        save(head_embedding, n_epoch);
                    }
                    printf("%d/%d epochs done\n", (n_epoch - 1), n_epochs);
                    hasNextEphos = false;
                }
            }
            checkForStop();
            if (stopped) break;
        }
        if (stopped) break;
    }
    free(epoch_of_next_negative_sample);
    free(epochs_per_negative_sample);
    free(epoch_of_next_sample);
}

int main(int argc, char **argv) {
    int i;
    if (argc<3){
        printf("Usage: %s <Input file> <Output file>", argv[0]);
        return 10;
    }
    int rand_seed = -1;
    if (argc==7){
        verbose=atoi(argv[6]);
    }
    if (argc ==6){
	    debugPrintLimit=atoi(argv[5]);
	    DEBUGGING=debugPrintLimit>0;
        if (argc<6){
            verbose=MAX_VERBOSE;
        }
    }
    if (verbose>0) {
        for (i = 0; i < argc; i++) {
            printf("Argument %i = %s\n", i, argv[i]);
        }
    }

    if (argc == 5) {
        save_progress = atoi(argv[4]);
        if (verbose>0)
            printf("save_progress passed as argument %d", save_progress);
    }

    if (argc == 4) {
        rand_seed = atoi(argv[3]);
        if (verbose>0)
            printf("rand_seed passed as argument %d", rand_seed);
    }

    if (rand_seed >= 0) {
        printf("Using random seed: %d\n", rand_seed);
        srand((unsigned int) rand_seed);
    } else {
        printf("Using current time as random seed...\n");
        srand(503);
    }

    char *inputFile;
    if (argc < 3) {
        outputFile = "result.txt";
        if (argc < 2) {
            inputFile = "data.txt";
        } else {
            inputFile = argv[1];
        }
    } else {
        outputFile = argv[2];
        inputFile = argv[1];
    }

    double **head_embedding, **tail_embedding;
    int *head, *tail;
    double *epochs_per_sample;
    int n_epochs, n_vertices, negative_sample_rate;
    double a, b, gamma, initial_alpha;
    int *randis;

    FILE *h;
    if ((h = fopen(inputFile, "r+b")) == NULL) {
        printf("Error: could not open data file.\n");
        return 20;
    }

    int temp;
    fscanf(h, "%d", &temp);
    const int n_components = temp;
    int size_head_embedding = 0;
    fscanf(h, "%d", &size_head_embedding);

    if (verbose>0)
        printf("Length of head_embedding is %d!\n", size_head_embedding);

    cols=n_components;
    rows=size_head_embedding;

    if (verbose>0)
        printf("n_components is %d!\n", n_components);

    head_embedding = create2DArray<double>(size_head_embedding, n_components);
    for (int i = 0; i < size_head_embedding; i++) {
        for (int j = 0; j < n_components; j++) {
            fscanf(h, "%lf", &head_embedding[i][j]);
        }
    }

    int size_tail_embedding = 0;
    fscanf(h, "%d", &size_tail_embedding);
    if (verbose>0)
        printf("Length of tail_embedding is %d!\n", size_tail_embedding);

    bool move_other;
    move_other = size_tail_embedding<1;
    if (verbose>0)
        printf("move_other: %d\n", move_other);

    if (move_other) {
        tail_embedding = head_embedding;
    } else {
        tail_embedding = create2DArray<double>(size_tail_embedding, n_components);
        printf("Size of head_embedding and tail_embedding is different");
        for (int i = 0; i < size_tail_embedding; i++) {
            for (int j = 0; j < n_components; j++) {
                fscanf(h, "%lf", &tail_embedding[i][j]);
            }
        }
    }
    int size_head = 0;
    fscanf(h, "%d", &size_head);
    if (verbose>0)
        printf("Length of head is %d!\n", size_head);
    head = (int *) malloc(size_head * sizeof(int));
    if (head == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    for (int i = 0; i < size_head; i++) {
        fscanf(h, "%d", &head[i]);
    }
    if (verbose>0)
        debug1D("head", head, size_head);
    int size_tail = 0;
    fscanf(h, "%d", &size_tail);
    if (verbose>0)
        printf("Length of tail is %d!\n", size_tail);

    tail = (int *) malloc(size_tail * sizeof(int));
    if (tail == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    for (int i = 0; i < size_tail; i++) {
        fscanf(h, "%d", &tail[i]);
    }
    if (verbose>0)
        debug1D("tail", tail, size_tail);

    int size_epochs = 0;
    fscanf(h, "%d", &size_epochs);
    if (verbose>>0)
        printf("Length of epochs is %d!\n", size_epochs);
    fscanf(h, "%d", &n_epochs);
    printf("n_epochs is %d \n", n_epochs);

    int size_n_vertices = 0;
    fscanf(h, "%d", &size_n_vertices);
    if (verbose>>0)
        printf("Length of n_vertices is %d!\n", size_n_vertices);
    fscanf(h, "%d", &n_vertices);
    printf("n_vertices is %d \n", n_vertices);

    int size_epochs_per_sample;
    fscanf(h, "%d", &size_epochs_per_sample);
    if (verbose>>0)
        printf("Length of epochs_per_sample is %d!\n", size_epochs_per_sample);
    epochs_per_sample = (double *) malloc(size_epochs_per_sample * sizeof(double));

    if (epochs_per_sample == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    for (int i = 0; i < size_epochs_per_sample; i++) {
        fscanf(h, "%lf", &epochs_per_sample[i]);
    }
    if (verbose>0)
        debug1D("epochs_per_sample", epochs_per_sample, size_epochs_per_sample);

    int size_a = 0;
    fscanf(h, "%d", &size_a);
    if (verbose>0)
        if (verbose>0)
            printf("Length of a is %d!\n", size_a);
    fscanf(h, "%lf", &a);
    if (verbose>0)
        printf("a value is  %lf \n", a);

    int size_b = 0;
    fscanf(h, "%d", &size_b);

    if (verbose>0)
        printf("Length of b is %d!\n", size_b);
    fscanf(h, "%lf", &b);

    if (verbose>0)
        printf("b values is  %lf \n", b);

    int size_gamma = 0;
    fscanf(h, "%d", &size_gamma);
    if (verbose>0)
        printf("Length of gamma is %d!\n", size_gamma);
    fscanf(h, "%lf", &gamma);
    if (verbose>0)
        printf("gamma value is %lf \n", gamma);

    int size_initial_alpha = 0;
    fscanf(h, "%d", &size_initial_alpha);
    if (verbose>0)
        printf("Length of initial_alpha is %d!\n", size_initial_alpha);
    fscanf(h, "%lf", &initial_alpha);
    if (verbose>0)
        printf("initial_alpha values is %lf \n", initial_alpha);

    int size_negative_sample_rate = 0;
    fscanf(h, "%d", &size_negative_sample_rate);
    if (verbose>0)
        printf("Length of negative_sample_rate is %d!\n", size_negative_sample_rate);
    fscanf(h, "%d", &negative_sample_rate);
    if (verbose>0)
        printf("negative_sample_rate value is %d \n", negative_sample_rate);

    int size_randis = 0;
    int debugRandomNumbers[] = {1069,754,674,957,46,714,415,1083,947,1070,298,1864};
    if (DEBUGGING) {
        // for purity of matching with Java we set the random numbers to a fixes list
        // within the expected 2000 event input file for samples2k.csv
        //CytoGate\src\edu\stanford\facs\swing\sgdInput.txt
        //... in MatLab we produced sgdInput.txt by running
        //	run_umap('sample2k.csv', 'method', 'C++');
        //	the sample2k.csv is in CVS folder CytoGate/matlabsrc/umap

            size_randis = 12;
            randis = debugRandomNumbers;
    }
    const int n_1_simplices = size_epochs_per_sample;
    fclose(h);

    if (verbose>0) {
        doVerbose(verbose, move_other, size_head_embedding, size_tail_embedding,
                head_embedding, tail_embedding, head, tail, n_epochs, n_vertices,
                epochs_per_sample, a, b, gamma, initial_alpha,
                negative_sample_rate, n_1_simplices, n_components);
    }else if (move_other){
        doMoveOther(head_embedding, tail_embedding, head, tail, n_epochs, n_vertices,
                    epochs_per_sample, a, b, gamma, initial_alpha,
                    negative_sample_rate, n_1_simplices, n_components);
    } else {
        doNotMoveOther(head_embedding, tail_embedding, head, tail, n_epochs, n_vertices,
                    epochs_per_sample, a, b, gamma, initial_alpha,
                    negative_sample_rate, n_1_simplices, n_components);
    }
    if (stopped)
        return -2;
    save(head_embedding, 0);

    //CLEANUP ... free() is not necessary in the main procedure since all memory is released upon exit
    //  ... so this is just an example of all the free() necessary when refactoring this to a class
    //      with a destructor methods
    if (verbose>0)
        printf("Free the memory\n");

    delete2DArray(head_embedding);
    head_embedding = NULL;
    if (!move_other) {
        delete2DArray(tail_embedding);
    }
     tail_embedding = NULL;
    free(head);
    head = NULL;
    free(tail);
    tail = NULL;
    free(epochs_per_sample);
    epochs_per_sample = NULL;

    //Success
    return 0;
}



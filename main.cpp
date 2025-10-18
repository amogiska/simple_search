#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include <chrono>

using namespace std;

// Read vectors from .fvecs binary file
// Format: [dim (4 bytes)] [float values (dim × 4 bytes)] repeated
vector<vector<float>> read_fvecs(const string& filename, int max_vectors = -1) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Cannot open file " << filename << endl;
        return {};
    }
    
    vector<vector<float>> vectors;
    int count = 0;
    
    while (file.peek() != EOF) {
        if (max_vectors > 0 && count >= max_vectors) {
            break;
        }
        
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!file) break;
        
        vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!file) break;
        
        vectors.push_back(vec);
        count++;
    }
    
    file.close();
    return vectors;
}

// Compute L2 (Euclidean) distance between two vectors
// Formula: sqrt(sum of (a[i] - b[i])^2)
float l2_distance(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size()) {
        cerr << "Error: Vectors have different dimensions!" << endl;
        return -1.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sqrt(sum);
}

// Search result: index and distance
struct SearchResult {
    int index;
    float distance;
    
    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }
    
    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }
};

// Brute force k-NN search: compares query to every vector in database
// Time Complexity: O(n * d) where n = vectors, d = dimensions
vector<SearchResult> brute_force_search(
    const vector<float>& query,
    const vector<vector<float>>& database,
    int k) {
    
    // Use max-heap to keep top k nearest neighbors
    priority_queue<SearchResult, vector<SearchResult>, less<SearchResult>> max_heap;
    
    cout << "\n[Search Progress]" << endl;
    cout << "Comparing query vector against " << database.size() << " vectors..." << endl;
    
    for (size_t i = 0; i < database.size(); i++) {
        float dist = l2_distance(query, database[i]);
        
        if (max_heap.size() < static_cast<size_t>(k)) {
            max_heap.push({static_cast<int>(i), dist});
        }
        else if (dist < max_heap.top().distance) {
            max_heap.pop();
            max_heap.push({static_cast<int>(i), dist});
        }
        
        if (database.size() >= 10 && (i + 1) % (database.size() / 10) == 0) {
            cout << "  Progress: " << (i + 1) << "/" << database.size() << " vectors" << endl;
        }
    }
    
    // Extract and reverse to get closest-first order
    vector<SearchResult> results;
    while (!max_heap.empty()) {
        results.push_back(max_heap.top());
        max_heap.pop();
    }
    reverse(results.begin(), results.end());
    
    return results;
}

int main(int argc, char* argv[]) {
    cout << "========================================" << endl;
    cout << "Brute Force Vector Search Demo" << endl;
    cout << "========================================" << endl;
    
    int NUM_BASE_VECTORS = 100;  // Default
    const int K = 10;
    
    if (argc > 1) {
        NUM_BASE_VECTORS = atoi(argv[1]);
        if (NUM_BASE_VECTORS <= 0) {
            cerr << "Error: Invalid number of vectors. Using default (100)." << endl;
            NUM_BASE_VECTORS = 100;
        }
    }
    
    cout << "\n[Step 1] Loading database vectors..." << endl;
    cout << "Reading up to " << NUM_BASE_VECTORS << " vectors from sift_base.fvecs" << endl;
    
    auto database = read_fvecs("sift_base.fvecs", NUM_BASE_VECTORS);
    
    if (database.empty()) {
        cerr << "Failed to load database vectors!" << endl;
        return 1;
    }
    
    int original_size = database.size();
    cout << " Loaded " << original_size << " vectors from file" << endl;
    
    // If requested more vectors than available, duplicate them
    if (NUM_BASE_VECTORS > original_size) {
        cout << " Requested " << NUM_BASE_VECTORS << " but only " << original_size << " available" << endl;
        cout << " Duplicating vectors to reach " << NUM_BASE_VECTORS << "..." << endl;
        
        while (database.size() < static_cast<size_t>(NUM_BASE_VECTORS)) {
            int idx = database.size() % original_size;
            database.push_back(database[idx]);
        }
        cout << " Database now has " << database.size() << " vectors (with duplicates)" << endl;
    }
    
    cout << " Each vector has " << database[0].size() << " dimensions" << endl;
    
    cout << "\n[Step 2] Loading query vector..." << endl;
    cout << "Reading first query from sift_query.fvecs" << endl;
    
    auto queries = read_fvecs("sift_query.fvecs", 1);
    
    if (queries.empty()) {
        cerr << "Failed to load query vector!" << endl;
        return 1;
    }
    
    vector<float> query = queries[0];
    cout << " Loaded query vector (dimension: " << query.size() << ")" << endl;
    
    cout << "  First 5 values: [";
    for (int i = 0; i < 5 && i < static_cast<int>(query.size()); i++) {
        cout << query[i];
        if (i < 4) cout << ", ";
    }
    cout << ", ...]" << endl;
    
    cout << "\n[Step 3] Performing brute force search..." << endl;
    cout << "Finding top " << K << " nearest neighbors" << endl;
    
    auto start_time = chrono::high_resolution_clock::now();
    auto results = brute_force_search(query, database, K);
    auto end_time = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    double seconds = duration.count() / 1000.0;
    
    cout << "\n[Step 4] Results!" << endl;
    cout << "========================================" << endl;
    cout << "Top " << K << " Nearest Neighbors:" << endl;
    cout << "Search Time: " << seconds << " seconds" << endl;
    cout << "========================================" << endl;
    
    for (size_t i = 0; i < results.size(); i++) {
        cout << "Rank " << (i + 1) << ": ";
        cout << "Vector #" << results[i].index;
        cout << " (distance: " << results[i].distance << ")";
        cout << endl;
    }
    
    return 0;
}

#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <list>
#include <fstream>
#include <cuda_runtime_api.h>

using namespace std;

const int SIZE = 60;
const int NEIGH = 128;
const int GAP_PEN = 2;
const int MIN_W = 10;

typedef char* seq;

vector<pair<seq, int>> data;
vector<pair<string, bool>> desc;
vector<pair<vector<int>, int>> keys;

char complementary(char c)
{
	switch(c) {
		case 'a':
			return 't';
		case 't':
			return 'a';
		case 'c':
			return 'g';
		case 'g':
			return 'c';
		default:
			return 'n';
	}
}

int toInt(char c)
{
	switch(c) {
		case 'a':
			return 0;
		case 't':
			return 3;
		case 'c':
			return 1;
		case 'g':
			return 2;
		default:
			return 4;
	}
}

struct comp
{
	inline bool operator()(const pair<seq, int>& a, const pair<seq, int>& b)
	{
		return keys[a.second].first < keys[b.second].first;
	}
};

void createCompl(int i)
{
	seq s = new char[SIZE];
	for (int j = 0; j < SIZE; j++) {
		s[SIZE-1-j] = complementary(data[i-1].first[j]);
	}
	data.push_back(make_pair(s,i));
}

void print_vec(const vector<int>& v)
{
	for (int x : v)
		cout<<x<<",";
	cout<<endl;
}
void createKey(int n)
{
	seq s = data[n].first;
	int trigrams[293];
	for (int i = 0; i<293; i++)
		trigrams[i] = 0;
	for (int i = 0; i < SIZE-2; i++) {
		int k = 0;
		for (int j = i; j < i+3 ; j++) {
			k+=toInt(s[j]);
			if (j < i+2)
				k = k << 3;
		}
		trigrams[k]++;
	}
	vector<int> tmp[SIZE-2];
	for (int i = 0; i < 293; i++) {
		if (trigrams[i] > 0) {
			tmp[trigrams[i]].push_back(i);
		}
	}
	vector<int> key;
	for (int i = SIZE-3; i >=0; i--) {
		if (tmp[i].size() > 0) {
			for (int x : tmp[i]) {
				for (int j = 0; j < i; j++) {
					key.push_back(x);
				}
			}
		}
	}
	keys.push_back(make_pair(key,n));
}

int NWalg(const seq s1, const seq s2, list<int> &align)
{
	int A[SIZE+1][SIZE+1];
	char B[SIZE+1][SIZE+1];
	for (int i = 0; i < SIZE+1; i++) {
		A[0][i] = 0;
		A[i][0] = 0;
	}
	for (int i = 0; i<SIZE+1;i++) {
		for (int j=0; j<SIZE+1;j++) {
			if (i > 0 && j > 0) {
				A[i][j] = (s1[i-1] == s2[j-1]) ? 1 : -2;
			}
			B[i][j] = 0;
		}
	}
	for (int i = 1; i<SIZE+1;i++) {
		for (int j=1; j<SIZE+1;j++) {
			int m = max(max(A[i-1][j-1]+A[i][j],A[i][j-1]-GAP_PEN),A[i-1][j]-GAP_PEN);
			if (m == A[i-1][j-1]+A[i][j]) {
				B[i][j] = 1;
			}
			else if (m == A[i][j-1]-GAP_PEN) {
				B[i][j] = 2;
			}
			else {
				B[i][j] = 3;
			}
			A[i][j] = m;
		}
	}
	int max_val = MIN_W-1;
	int max_i, max_j;
	max_i = -1;
	max_j = -1;
	for (int i = 0; i<SIZE+1; i++) {
		if (A[SIZE][i] > max_val) {
			max_i = SIZE;
			max_j = i;
			max_val = A[SIZE][i];
		}
		else if (A[i][SIZE] > max_val) {
			max_i = i;
			max_j = SIZE;
			max_val = A[i][SIZE];
		}
	}
	while (max_i > 0 && max_j > 0) {
		if (B[max_i][max_j] == 1) {
			max_i--;
			max_j--;
			align.push_front(max_i-1);
			align.push_front(max_j-1);
		}
		else if (B[max_i][max_j] == 2) {
			max_j--;
		}
		else {
			max_i--;
		}
	}
	return (max_val >= MIN_W) ? max_val : 0;
}

__device__ int NWalg(const seq s1, const seq s2, char* alignment)
{
	int A[SIZE+1][SIZE+1];
	char B[SIZE+1][SIZE+1];
	for (int i = 0; i < SIZE+1; i++) {
		A[0][i] = 0;
		A[i][0] = 0;
	}
	for (int i = 0; i<SIZE+1;i++) {
		for (int j=0; j<SIZE+1;j++) {
			if (i > 0 && j > 0) {
				A[i][j] = (s1[i-1] == s2[j-1]) ? 1 : -2;
			}
			B[i][j] = 0;
		}
	}
	for (int i = 1; i<SIZE+1;i++) {
		for (int j=1; j<SIZE+1;j++) {
			int m = max(max(A[i-1][j-1]+A[i][j],A[i][j-1]-GAP_PEN),A[i-1][j]-GAP_PEN);
			if (m == A[i-1][j-1]+A[i][j]) {
				B[i][j] = 1;
			}
			else if (m == A[i][j-1]-GAP_PEN) {
				B[i][j] = 2;
			}
			else {
				B[i][j] = 3;
			}
			A[i][j] = m;
		}
	}
	int max_val = MIN_W-1;
	int max_i, max_j;
	max_i = -1;
	max_j = -1;
	for (int i = 0; i<SIZE+1; i++) {
		if (A[SIZE][i] > max_val) {
			max_i = SIZE;
			max_j = i;
			max_val = A[SIZE][i];
		}
		else if (A[i][SIZE] > max_val) {
			max_i = i;
			max_j = SIZE;
			max_val = A[i][SIZE];
		}
	}

	if (max_val < MIN_W) {
		return 0;
	}

	while (max_i > 0 && max_j > 0) {
		if (B[max_i][max_j] == 1) {
			alignment[max_i-1] = max_j-1;
			max_i--;
			max_j--;
		}
		else if (B[max_i][max_j] == 2) {
			max_j--;
		}
		else {
			max_i--;
		}
	}
	return (max_val >= MIN_W) ? max_val : 0;
}

__global__ void gpualgorithm(const char* sequences, int* res, char* res_align, const int seqNum)
{
	__shared__ char data[SIZE*(NEIGH+1)];
	__shared__ int master;
	__shared__ int seqIdx;
	__shared__ int round;
	int ret;
	char* alignment;

	alignment = new char[SIZE];

	int idx = threadIdx.x;

	if (idx == 0) {
		master = 0;
		seqIdx = NEIGH + 1;
		round = 0;
	}

	for (int i = 0; i < SIZE; i++) {
		data[idx*SIZE+i] = sequences[idx*SIZE+i];
	}

	__syncthreads();

	while (master <= NEIGH || seqIdx <= seqNum) {
		//#pragma unroll
		for (int i = 0; i < SIZE; i++) {
			alignment[i] = -1;
		}

		if (idx != master) {
			ret = NWalg(data + master*SIZE, data + idx*SIZE, alignment);
			int ridx = master * NEIGH + round * (NEIGH + 1) * NEIGH; 
			if (idx > master) {
				ridx += idx;
				ridx = ridx - master - 1;
			}
			else {
				ridx += NEIGH + idx;
				ridx -= master;
			}
			res[ridx] = ret;
			if (ret > 0) {
				ridx*=SIZE;
				//#pragma unroll
				for (int i = 0; i < SIZE; i++) {
					res_align[ridx + i] = alignment[i];		
				}
			}
		}

		__syncthreads();

		//copying new seqence into data buffer
		if (idx < SIZE && seqIdx < seqNum) {
			data[master*SIZE + idx] = sequences[seqIdx*SIZE + idx];
		}

		__syncthreads();

		if (idx == master) {
			if (seqIdx <= seqNum) {
				seqIdx++;
				master = (master + 1)%(NEIGH + 1);
			}
			else {
				master++;
			}
			if (master == 0) {
				round++;
			}
		}

		__syncthreads();
	}
}

int main(int argc, const char** argv)
{
	char* sequences, *sequencesDev, *res_align, *res_alignDev;
	int *res, *resDev;
	cudaError_t status;
	int i = 0;
	if (argc != 3) {
		cout<<"Usage: 2 arguments: input file output file\n";
		return 0;
	}
	ifstream input;
	input.open(argv[1]);
	ofstream output;
	output.open(argv[2]);
	char x;
	input>>x;                                                                      
	string str;
	while(getline(input,str)) {                                                
		desc.push_back(make_pair(str, 0));
		char x;                                                                   
		seq s = new char[SIZE];
		input.getline(s,SIZE+1);
		data.push_back(make_pair(s,i));                                                  
		i++;                                                                      
		//creating complementary sequences
		desc.push_back(make_pair(desc[i-1].first, 1));
		createCompl(i);
		i++;
		input>>x;
	}
	for (int i = 0; i < data.size(); i++) {
		createKey(i);
	}
	comp c;
	sort(begin(data), end(data), c);
	sequences = new char[SIZE*data.size()];
	res = new int[NEIGH*data.size()];
	res_align = new char[SIZE * NEIGH*data.size()];
	for (size_t i = 0; i < data.size(); i++) {
		for (size_t j = 0; j < SIZE; j++) {
			sequences[i*SIZE + j] = data[i].first[j];
		}
	}
	status = cudaMalloc(&sequencesDev, data.size() * SIZE * sizeof(char));
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	status = cudaMalloc(&resDev, data.size() * NEIGH * sizeof(int));
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	status = cudaMalloc(&res_alignDev, data.size() * SIZE * NEIGH * sizeof(char));
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}
	status = cudaMemcpy(sequencesDev, sequences, data.size() * SIZE * sizeof(char), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	gpualgorithm<<<1, NEIGH+1>>>(sequencesDev, resDev, res_alignDev, data.size());

	status = cudaMemcpy(res, resDev, data.size() * NEIGH * sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	status = cudaMemcpy(res_align, res_alignDev, data.size() * SIZE * NEIGH * sizeof(char), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < NEIGH; j++) {
			if (i+j+1 < data.size()) {
				int r = res[i*NEIGH + j];
				if (r > 0) {
					output<<desc[data[i].second].first<<"; "<< desc[data[i+j+1].second].first<<"; ";
					output<<r<<"; ";
					output<<((desc[data[i].second].second) ? "1," : "0,");
					output<<((desc[data[i+j+1].second].second) ? " 1" : " 0");
					output<<"; {";
					bool colon = false;
					for (int k = 0; k < SIZE; k++) {
						if ((int)res_align[i*SIZE*NEIGH + j*SIZE +k] != -1) {
							if (colon) {
								output<<";";
							}
							output<<k<<"-"<<(int)res_align[i*SIZE*NEIGH + j*SIZE + k];
							colon = true;
						}
					}
					/*
						for (auto it = align.begin(); it != align.end(); it++) {
						if (x % 2 == 0 && it != align.begin())
						output<<";";
						output<<*it+1;
						if (x % 2 == 0)
						output<<"-";
						x++;
						}
					 */
					output<<"}";
					output<<endl;

				}
			}
		}
	}
	return 0;
}

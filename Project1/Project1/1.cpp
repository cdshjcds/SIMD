#include<iostream>
#include<immintrin.h>
#include<cmath>
#include<time.h>
#include<fstream>

#define max_N 10000

using namespace std;

void avx(int m,int n,float a[][max_N]) {
	__m256 t1, t2, t3, c;
	for (int k = 0; k < (m < n ? m : n); k++)
	{
		
		//选取k列绝对值最大的元素为主元,主元在row行
		float s = abs(a[k][k]);
		int row = k;
		for (int i = k+1; i < m; i++)
		{
			float t = abs(a[i][k]);
			if (s < t)
			{
				s = t;
				row = i;
			}
		}
		//k列全为0则不需处理
		if (s == 0)
			continue;
		//交换row行与k行
		if (row != k) 
			for (int j = 0; j < n; j++)
				swap(a[k][j], a[row][j]);
		for (int j = k+1; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1;
		for (int j = k + 1; j <= n-8; j+=8)
		{
			t1 = _mm256_loadu_ps(a[k]+j);
			for (int i = k + 1; i < m; i++)
			{
				t2 = _mm256_loadu_ps(a[i] + j);
				c = _mm256_set1_ps(a[i][k]);
				t3 = _mm256_mul_ps(t1, c);
				t2 = _mm256_sub_ps(t2,t3);
				_mm256_store_ps(a[i]+j,t2);
			}
		}
		for (int j = n - (n - k-1) % 8; j < n; j++)
			for (int i = k + 1; i < m; i++)
				a[i][j] = a[i][j] - a[i][k]*a[k][j];
		for (int i = k + 1; i < m; i++)
			a[i][k] = 0;
	}
}

void avx2(int m, int n, float a[][max_N]) {
	__m256 t1, t2, t3, c;
	int T = 32;
	for (int k = 0; k < (m < n ? m : n); k++)
	{
		//选取k列绝对值最大的元素为主元,主元在row行
		float s = abs(a[k][k]);
		int row = k;
		for (int i = k + 1; i < m; i++)
		{
			float t = abs(a[i][k]);
			if (s < t)
			{
				s = t;
				row = i;
			}
		}
		//k列全为0则不需处理
		if (s == 0)
			continue;
		//交换row行与k行
		if (row != k)
			for (int j = 0; j < n; j++)
				swap(a[k][j], a[row][j]);
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1;
		for (int t = 0; t < ((m - k - 1) % T > 0 ? 1 + (m - k - 1) / T : (m - k - 1) / T); t++)
		{
			for (int j = k + 1; j <= n - 8; j += 8)
			{
				t1 = _mm256_loadu_ps(a[k] + j);
				for (int i = k + 1 + t * T; i < (m > k + 1 + t * T + T ? k + 1 + t * T + T : m); i++)
				{
					t2 = _mm256_loadu_ps(a[i] + j);
					c = _mm256_set1_ps(a[i][k]);
					t3 = _mm256_mul_ps(t1, c);
					t2 = _mm256_sub_ps(t2, t3);
					_mm256_store_ps(a[i] + j, t2);
				}
			}
		}
		for (int j = n - (n - k - 1) % 8; j < n; j++)
			for (int i = k + 1; i < m; i++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (int i = k + 1; i < m; i++)
			a[i][k] = 0;
	}
}

void avx3(int m, int n, float a[][max_N]) {
	__m256 t, c;
	for (int k = 0; k < (m < n ? m : n); k++)
	{

		//选取k列绝对值最大的元素为主元,主元在row行
		float s = abs(a[k][k]);
		int row = k;
		for (int i = k + 1; i < m; i++)
		{
			float t = abs(a[i][k]);
			if (s < t)
			{
				s = t;
				row = i;
			}
		}
		//k列全为0则不需处理
		if (s == 0)
			continue;
		//交换row行与k行
		if (row != k)
			for (int j = 0; j < n; j++)
				swap(a[k][j], a[row][j]);
		c = _mm256_set1_ps(a[k][k]);
		for (int j = k + 1; j <= n - 8; j += 8)
		{
			t = _mm256_loadu_ps(a[k] + j);
			t = _mm256_div_ps(t, c);
			_mm256_store_ps(a[k] + j, t);
		}
		for (int j = n - (n - k - 1) % 8; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1;
		for (int j = k + 1; j < n; j++)
			for (int i = k + 1; i < m; i++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (int i = k + 1; i < m; i++)
			a[i][k] = 0;
	}
}

void normal(int m, int n, float a[][max_N]) {
	for (int k = 0; k < (m < n ? m : n); k++)
	{

		//选取k列绝对值最大的元素为主元,主元在row行
		float s = abs(a[k][k]);
		int row = k;
		for (int i = k + 1; i < m; i++)
		{
			float t = abs(a[i][k]);
			if (s < t)
			{
				s = t;
				row = i;
			}
		}
		//k列全为0则不需处理
		if (s == 0)
			continue;
		//交换row行与k行
		if (row != k)
			for (int j = 0; j < n; j++)
				swap(a[k][j], a[row][j]);
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1;
		for (int j = k+1; j < n; j++)
			for (int i = k + 1; i < m; i++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (int i = k + 1; i < m; i++)
			a[i][k] = 0;
	}
}

void normal2(int m, int n, float a[][max_N]) {
	for (int k = 0; k < (m < n ? m : n); k++)
	{

		//选取k列绝对值最大的元素为主元,主元在row行
		float s = abs(a[k][k]);
		int row = k;
		for (int i = k + 1; i < m; i++)
		{
			float t = abs(a[i][k]);
			if (s < t)
			{
				s = t;
				row = i;
			}
		}
		//k列全为0则不需处理
		if (s == 0)
			continue;
		//交换row行与k行
		if (row != k)
			for (int j = 0; j < n; j++)
				swap(a[k][j], a[row][j]);
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1;
		for (int j = k + 1; j % 4 != 0; j++)
			for (int i = k + 1; i < m; i++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (int j = k + 1 + (4 - (k + 1) % 4) % 4; j < n; j++)
			for (int i = k + 1; i < m; i++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (int i = k + 1; i < m; i++)
			a[i][k] = 0;
	}
}

float A[max_N][max_N];

int main()
{
	ifstream in("data.txt");
	if (!in.is_open()) {
		cerr << "Failed to open file!" << endl;
		return -1;
	}
	int m, n;
	in >> m ;
	in >> n;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			in >> A[i][j];
		}
	}
	time_t start, end;
	start = clock();
	avx2(m, n, A);
	end = clock();
	/*
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			cout << A[i][j]<<" ";
		cout << endl;
	}*/
	cout << "Time useage: " << (float)(end - start)/1000 << " s" << endl;
}
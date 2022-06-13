#include <cstdio>
#include <string.h> // for memcpy
// 共通部分式除去に期待する
#define sq(x) ((x) * (x))

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;

const double dx = 2 / ((double)(nx - 1));
const double dy = 2 / ((double)(ny - 1));
const double dt = 0.01;
const double rho = 1;
const double nu = 0.02;

double x[nx], y[ny]; // init later

// init with zero
double u[nx][ny];
double v[nx][ny];
double p[nx][ny];
double b[nx][ny];

double X[nx][ny], Y[nx][ny]; // init later

void init()
{
  // x = np.linespace(0, 2, nx)
  for (int i = 0; i < nx; i++)
    x[i] = (2 - 0) * i / nx;
  // y = np.linespace(0, 2, ny)
  for (int i = 0; i < ny; i++)
    y[i] = (2 - 0) * i / ny;

  // X, Y = np.meshgrid(x, y)
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
    {
      X[i][j] = x[i];
      Y[i][j] = y[j];
    }
}

void print(double u[nx][ny], double v[nx][ny], int count)
{
  printf("----%2d----\n\n", count);

  printf("----u----\n");
  for (int j = 0; j < ny; j++)
  {
    for (int i = 0; i < nx - 1; i++)
    {
      printf("%2.1e ", u[j][i]);
    }
    printf("%2.1e\n", u[j][nx - 1]);
  }

  printf("----v----\n");
  for (int j = 0; j < ny; j++)
  {
    for (int i = 0; i < nx - 1; i++)
    {
      printf("%2.1e ", v[j][i]);
    }
    printf("%2.1e\n", v[j][nx - 1]);
  }
  printf("----------\n\n");
}

int main()
{
  init();

  for (int n = 0; n < nt; n++)
  {
    for (int j = 1; j < ny - 1; j++)
    {
      for (int i = 1; i < nx - 1; i++)
      {
        b[j][i] = rho * (1 / dt *
                             ((u[j][i + 1] - u[j][i - 1]) / ((2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy))) -
                             sq((u[j][i + 1] - u[j][i - 1]) / (2 * dx)) - 2 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy) *
                                (v[j][i + 1] - v[j][i - 1]) / (2 * dx)) - sq((v[j + 1][i] - v[j - 1][i]) / (2 * dy)));
      }
    }
    // ループ間に依存関係がない
    #pragma openmp parallel for
    for (int it = 0; it < nit; it++)
    {
      double pn[nx][ny];
      memcpy(pn, p, nx * ny * sizeof(double));
      for (int j = 1; j < ny - 1; j++)
      {
        for (int i = 1; i < nx - 1; i++)
        {
          p[j][i] = (sq(dy) * (pn[j][i + 1] + pn[j][i - 1]) +
                     sq(dx) * (pn[j + 1][i] + pn[j - 1][i]) -
                     b[j][i] * sq(dx) * sq(dy)) /
                    (2 * (sq(dx) + sq(dy)));
        }
      }
      for (int i = 0; i < nx; i++)
        p[i][ny - 1] = p[i][ny - 2];
      for (int j = 0; j < ny; j++)
        p[0][j] = p[1][j];
      for (int i = 0; i < nx; i++)
        p[i][0] = p[i][1];
      for (int j = 0; j < ny; j++)
        p[nx - 1][j] = 0;
    }

    double un[nx][ny], vn[nx][ny];
    memcpy(un, u, nx * ny * sizeof(double));
    memcpy(vn, v, nx * ny * sizeof(double));
    for (int j = 1; j < ny - 1; j++)
    {
      for (int i = 1; i < nx - 1; i++)
      {
        u[j][i] = un[j][i] -
                  un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
                  un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
                  dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) +
                  nu * dt / sq(dx) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) +
                  nu * dt / sq(dy) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
        v[j][i] = vn[j][i] -
                  vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
                  vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
                  dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) +
                  nu * dt / sq(dx) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) +
                  nu * dt / sq(dy) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
      }
    }
    for (int i = 0; i < nx; i++)
      u[i][0] = 0;
    for (int j = 0; j < ny; j++)
      u[0][j] = 0;
    for (int i = 0; i < nx; i++)
      u[i][ny - 1] = 0;
    for (int j = 0; j < ny; j++)
      u[nx - 1][j] = 1;
    for (int i = 0; i < nx; i++)
      v[i][0] = 0;
    for (int j = 0; j < ny; j++)
      v[0][j] = 0;
    for (int i = 0; i < nx; i++)
      v[i][ny - 1] = 0;
    for (int j = 0; j < ny; j++)
      v[nx - 1][j] = 0;
    print(u, v, n);
  } // for n in nt
} // main

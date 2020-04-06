#include <iostream>
#include <stdio.h>
using namespace std;


int input[1000][1000];
int kernel[50][50];
int output[1000][1000];

int main()
{

    freopen("data.txt", "r", stdin);
    int Row = 5, Col = 5;
    int kr = 2, kc = 2;
    for(int i=0; i<Row; i++)
        for(int j=0;j<Col; j++)
            cin>>input[i][j];
    
    for(int i=0; i<kr; i++)
        for(int j=0;j<kc; j++)
            cin>>kernel[i][j];

    int outputRow = Row - 1;
    int outputCol = Col - 1;

    for(int i=0; i+(kr-1)<Row; i++)
    {
        for(int j=0; j+(kc-1)<Col; j++)
        {
            int sum = 0;
            for(int k = i, m = 0; m<kr; m++, k++)
            {
                for(int l = j, n = 0; n<kc; n++, l++)
                {   cout<<input[k][l]<<" x "<< kernel[m][n]<<endl;
                    sum = sum + input[k][l] * kernel[m][n];
                }
            }
            output[i][j] = sum;
            cout<<endl;
            cout<<endl;
        }
    }

    for(int i=0; i<outputRow; i++)
    {
        for(int j=0;j<outputCol; j++)
            cout<<output[i][j]<<" ";
        cout<<endl;
    }

    return 0;
}
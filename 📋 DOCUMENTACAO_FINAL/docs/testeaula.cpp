#include<stdio.h>

int main(){ 
    int num=1;
    int soma=0;


    while (num<1000){
    if (num%2==0){
        soma = soma+num;
    }
    num++;c
    }
    printf("A soma dos numeros pares menores que 1000 é %d\n",soma);
    

    return 0;
}

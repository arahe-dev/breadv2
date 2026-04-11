#include <stdio.h>
#include <unistd.h>

int main() {
    // Test input for server mode
    printf("<|im_start|>system\n");
    printf("You are a helpful assistant.\n");
    printf("<|im_end|>\n");
    printf("<|im_start|>user\n");
    printf("What is 2+2?\n");
    printf("<|im_end|>\n");
    printf("<|im_start|>assistant\n");
    fflush(stdout);

    // Keep stdin open
    sleep(120);
    return 0;
}

#include <assert.h>
#include <stdio.h>

int main() {
   printf("Running tests...\n");

   // Dummy test: 1 + 1 == 2
   int result = 1 + 1;
   assert(result == 2);

   printf("All tests passed!\n");
   return 0;
}
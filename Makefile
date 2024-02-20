example:
	clang++ -std=c++17 -O0 -g -Wall main.cpp -o example;
	./example;
prof:
	clang++ -std=c++17 -O0 -pg -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -Wall main.cpp -o prof
quick:
	clang++ -std=c++17 -O3 -march=native -Wall quick.cpp -o quick;
	./quick;
	rm quick;
quick_man:
	clang++ -std=c++17 -O3 -march=native -Wall -DNO_MANUAL_VECTORIZATION quick.cpp -o quick;
	./quick;
	rm quick;
quick_noarch:
	clang++ -std=c++17 -O3 -Wall quick.cpp -o quick;
	./quick;
	rm quick;
clean:
	rm -rf example quick prof

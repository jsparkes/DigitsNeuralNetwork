case $(uname -s) in
Darwin)
	g++ -std=c++17 NeuralNetwork.cpp main.cpp pngwriter.cc -I../../eigen -I/usr/include/freetype2 -I/usr/local/include/freetype2 -lpng16 -lfreetype
	;;
*)
	g++ NeuralNetwork.cpp main.cpp pngwriter.cc -I../../eigen -I/usr/include/freetype2 -lpng16 -lfreetype
	;;
esac

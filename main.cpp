// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
// #include <io.h>
#include <filesystem>
#include <time.h>

#include "NeuralNetwork.h"
#include "pngwriter.h"

#define CMD_RESET	"\x1b[0m"
#define CMD_RED		"\x1b[31;1m"
#define CMD_GREEN	"\x1b[32;1m"
// main.cpp 

void readPng(const char* filepath, RowVector*& data) {
	pngwriter image;
	image.readfromfile(filepath);
	int width = image.getwidth();
	int height = image.getheight();
	data = new RowVector(width * height);

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			data->coeffRef(0, y * width + x) = image.dread(x, y);
}

bool train(NeuralNetwork& net, const char* file, int desired, int serial, bool testing) {

	RowVector* input;
	readPng(file, input);

	RowVector output(10);
	for (int c = 0; c < 10; c++)
		output.coeffRef(0, c) = c == desired ? 1 : 0;
	if (testing) {
		net.test(*input, output);
		net.evaluate(output);
	} else
		net.train(*input, output);
	delete input;

	double value = 0;
	int actual = net.vote(value);
	cout << serial << ' ' << file << " >> ";
	if (desired != actual)
		cout << CMD_RED << actual << CMD_RESET " (" << value - net.output(desired) << ")" << endl;
	else
		cout << CMD_GREEN << actual << CMD_RESET << endl;

	return desired == actual;
}

void enumFolder(const char* folder, vector<string>& files) {
	string str = folder, pathName;
	str.append("\\*.png");

	for (const auto& entry : std::filesystem::directory_iterator(folder))
	{
		if (std::filesystem::path(entry.path()).extension() == ".png")
		{
			// files.push_back(string(folder) + "/" + entry.path().generic_string());
			files.push_back(entry.path().generic_string());
		}
	}
}

#define DIGITS 10

void train(NeuralNetwork& net) {
	vector<string> files[DIGITS];
	vector<string>::iterator it[DIGITS];
	// read files names
	for (int n = 0; n < DIGITS; n++) {
		string path = ("mnist-pngs/train/") + string(1, '0' + n);
		enumFolder(path.c_str(), files[n]);
	}

	double cost = 0;
	int serial = 0, success = 0;
	// tain three times for better accuracy
	for (int trial = 0; trial < 3; trial++) {
		for (int n = 0; n < DIGITS; n++)
			it[n] = files[n].begin();
		while (true) {
			int run = 0;
			for (int n = 0; n < DIGITS; n++)
				if (it[n] != files[n].end()) {
					if (train(net, (it[n]++)->c_str(), n, ++serial, false))
						success++;
					cost += net.mse();
					run++;
				}
			if (run != DIGITS)
				break;
		}
	}
	cout << "TRAINING:" << endl;
	cout << "cost\t" << cost / (serial * 2) << endl;
	cout << "success\t" << success << endl;
	cout << "fail\t" << serial - success << endl;
	cout << "error\t" << (double)(serial - success) / serial * 100 << "%" << endl;
	cout << endl;
	
	net.save("params.txt");
}

void test(NeuralNetwork& net) {
	double cost = 0;
	int serial = 0, success = 0;
	vector<string> files[DIGITS];
	vector<string>::iterator it[DIGITS];
	for (int n = 0; n < DIGITS; n++) {
		string path = ("mnist-pngs/test/") + string(1, '0' + n);
		enumFolder(path.c_str(), files[n]);
	}

	for (int n = 0; n < DIGITS; n++)
		it[n] = files[n].begin();
	while (true) {
		int run = 0;
		for (int n = 0; n < DIGITS; n++)
			if (it[n] != files[n].end()) {
				if (train(net, (it[n]++)->c_str(), n, ++serial, true))
					success++;
				cost += net.mse();
				run++;
			}
		if (run == 0)
			break;
	}
	cout << "TESTING:" << endl;
	cout << "cost\t" << cost / (serial * 2) << endl;
	cout << "success\t" << success << endl;
	cout << "fail\t" << serial - success << endl;
	cout << "error\t" << (double)(serial - success) / serial * 100 << "%" << endl;
	cout << endl;
}

void evaluate(NeuralNetwork& net) {
	RowVector* precision, * recall;
	net.confusionMatrix(precision, recall);

	double precisionVal = precision->sum() / precision->cols();
	double recallVal = recall->sum() / recall->cols();
	double f1score = 2 * precisionVal * recallVal / (precisionVal + recallVal);

	cout << "Confusion matrix:" << endl;
	cout << *net.mConfusion << endl;
	cout << "Precision: " << (int)(precisionVal * 100) << '%' << endl;
	cout << *precision << endl;
	cout << "Recall: " << (int)(recallVal * 100) << '%' << endl;
	cout << *recall << endl;
	cout << "F1 score: " << (int)(f1score * 100) << '%' << endl;
	delete precision;
	delete recall;
}

int main() {
	NeuralNetwork net;
	if (!net.load("params.txt")) {
		net.init({ 28 * 28, 64, 16, 10 }, 0.05, NeuralNetwork::Activation::SIGMOID);
		train(net);
	}
	test(net);
	evaluate(net);

	return 0;
}

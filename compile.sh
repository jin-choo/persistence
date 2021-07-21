echo compiling java sources...
rm -rf class
mkdir class

javac -cp ./lib/fastutil-8.4.2.jar:./lib/commons-math3-3.6.1.jar -d class $(find ./src -name *.java)

echo make jar archive
cd class
jar cf persistence-1.0.jar ./
rm ../persistence-1.0.jar
mv persistence-1.0.jar ../
cd ..
rm -rf class

echo done.

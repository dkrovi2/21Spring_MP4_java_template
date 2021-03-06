\rm -rf OrphanPagesClasses C-output
mkdir ./OrphanPagesClasses
javac -cp $(hadoop classpath) OrphanPages.java -d OrphanPagesClasses
jar -cvf OrphanPages.jar -C OrphanPagesClasses/ ./
hadoop jar OrphanPages.jar OrphanPages dataset/links ./C-output
head C-output/part-r-00000
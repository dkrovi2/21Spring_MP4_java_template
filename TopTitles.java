import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class TopTitles extends Configured implements Tool {

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new TopTitles(), args);
    System.exit(res);
  }

  @Override
  public int run(String[] args) throws Exception {
    Configuration conf = this.getConf();
    FileSystem fs = FileSystem.get(conf);
    Path tmpPath = new Path("./tmp");
    fs.delete(tmpPath, true);

    Job jobA = Job.getInstance(conf, "Title Count");
    jobA.setOutputKeyClass(Text.class);
    jobA.setOutputValueClass(IntWritable.class);

    jobA.setMapperClass(TitleCountMap.class);
    jobA.setReducerClass(TitleCountReduce.class);

    FileInputFormat.setInputPaths(jobA, new Path(args[0]));
    FileOutputFormat.setOutputPath(jobA, tmpPath);

    jobA.setJarByClass(TopTitles.class);
    jobA.waitForCompletion(true);

    Job jobB = Job.getInstance(conf, "Top Titles");
    jobB.setOutputKeyClass(Text.class);
    jobB.setOutputValueClass(IntWritable.class);

    jobB.setMapOutputKeyClass(NullWritable.class);
    jobB.setMapOutputValueClass(TextArrayWritable.class);

    jobB.setMapperClass(TopTitlesMap.class);
    jobB.setReducerClass(TopTitlesReduce.class);
    jobB.setNumReduceTasks(1);

    FileInputFormat.setInputPaths(jobB, tmpPath);
    FileOutputFormat.setOutputPath(jobB, new Path(args[1]));

    jobB.setInputFormatClass(KeyValueTextInputFormat.class);
    jobB.setOutputFormatClass(TextOutputFormat.class);

    jobB.setJarByClass(TopTitles.class);
    return jobB.waitForCompletion(true) ? 0 : 1;
  }

  public static String readHDFSFile(String path, Configuration conf) throws IOException {
    Path pt = new Path(path);
    FileSystem fs = FileSystem.get(pt.toUri(), conf);
    FSDataInputStream file = fs.open(pt);
    BufferedReader buffIn = new BufferedReader(new InputStreamReader(file));

    StringBuilder everything = new StringBuilder();
    String line;
    while ((line = buffIn.readLine()) != null) {
      everything.append(line);
      everything.append("\n");
    }
    return everything.toString();
  }

  public static class TextArrayWritable extends ArrayWritable {
    public TextArrayWritable() {
      super(Text.class);
    }

    public TextArrayWritable(String[] strings) {
      super(Text.class);
      Text[] texts = new Text[strings.length];
      for (int i = 0; i < strings.length; i++) {
        texts[i] = new Text(strings[i]);
      }
      set(texts);
    }
  }

  public static class TitleCountMap extends Mapper<Object, Text, Text, IntWritable> {
    private static final IntWritable one = new IntWritable(1);
    private Text text = new Text();

    List<String> stopWords;
    String delimiters;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {

      Configuration conf = context.getConfiguration();

      String stopWordsPath = conf.get("stopwords");
      String delimitersPath = conf.get("delimiters");

      this.stopWords = Arrays.asList(readHDFSFile(stopWordsPath, conf).split("\n"));
      this.delimiters = readHDFSFile(delimitersPath, conf);
    }

    @Override
    public void map(Object key, Text value, Context context)
        throws IOException, InterruptedException {
      StringTokenizer st = new StringTokenizer(value.toString(), delimiters);
      while (st.hasMoreTokens()) {
        String word = st.nextToken().toLowerCase(Locale.ROOT);
        if (!stopWords.contains(word)) {
          text.set(word);
          context.write(text, one);
        }
      }
    }
  }

  public static class TitleCountReduce extends Reducer<Text, IntWritable, Text, IntWritable> {
    IntWritable result = new IntWritable();

    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      int count = 0;
      for (IntWritable val : values) {
        count += val.get();
      }
      result.set(count);
      context.write(key, result);
    }
  }

  public static class TopTitlesMap extends Mapper<Text, Text, NullWritable, TextArrayWritable> {
    int n;
    private final TreeSet<Pair<Integer, String>> titlesByCount = new TreeSet<Pair<Integer, String>>();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      n = conf.getInt("N", 10);
    }

    @Override
    public void map(Text key, Text value, Context context)
        throws IOException, InterruptedException {
      Integer count = Integer.parseInt(value.toString());
      String word = key.toString();

      titlesByCount.add(new Pair<>(count, word));
      if (titlesByCount.size() > n) {
        titlesByCount.remove(titlesByCount.first());
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      for (Pair<Integer, String> item: titlesByCount) {
        String[] strings = {item.second, item.first.toString()};
        TextArrayWritable val = new TextArrayWritable(strings);
        context.write(NullWritable.get(), val);
      }
    }
  }

  public static class TopTitlesReduce extends Reducer<NullWritable, TextArrayWritable, Text, IntWritable> {
    private final TreeSet<Pair<Integer, String>> titlesByCount = new TreeSet<>();
    private int n;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      this.n = conf.getInt("N", 10);
    }

    @Override
    public void reduce(NullWritable key, Iterable<TextArrayWritable> values, Context context) throws IOException, InterruptedException {
      for (TextArrayWritable val: values) {
        Text[] pair= (Text[]) val.toArray();
        String word = pair[0].toString();
        Integer count = Integer.parseInt(pair[1].toString());
        titlesByCount.add(new Pair<>(count, word));
        if (titlesByCount.size() > n) {
          titlesByCount.remove(titlesByCount.first());
        }
      }

      for (Pair<Integer, String> item: titlesByCount) {
        Text word = new Text(item.second);
        IntWritable value = new IntWritable(item.first);
        context.write(word, value);
      }
    }
  }

  private static class Pair<A extends Comparable<? super A>, B extends Comparable<? super B>>
      implements Comparable<Pair<A, B>> {

    public final A first;
    public final B second;

    public Pair(A first, B second) {
      this.first = first;
      this.second = second;
    }

    public static <A extends Comparable<? super A>, B extends Comparable<? super B>> Pair<A, B> of(
        A first, B second) {
      return new Pair<A, B>(first, second);
    }

    @Override
    public int compareTo(Pair<A, B> o) {
      int cmp = o == null ? 1 : (this.first).compareTo(o.first);
      return cmp == 0 ? (this.second).compareTo(o.second) : cmp;
    }

    @Override
    public int hashCode() {
      return 31 * hashcode(first) + hashcode(second);
    }

    private static int hashcode(Object o) {
      return o == null ? 0 : o.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof Pair)) return false;
      if (this == obj) return true;
      return equal(first, ((Pair<?, ?>) obj).first) && equal(second, ((Pair<?, ?>) obj).second);
    }

    private boolean equal(Object o1, Object o2) {
      return o1 == o2 || (o1 != null && o1.equals(o2));
    }

    @Override
    public String toString() {
      return "(" + first + ", " + second + ')';
    }
  }
}

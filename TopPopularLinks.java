import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
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

import java.io.IOException;
import java.lang.Integer;
import java.util.StringTokenizer;
import java.util.TreeSet;

public class TopPopularLinks extends Configured implements Tool {
  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new TopPopularLinks(), args);
    System.exit(res);
  }

  @Override
  public int run(String[] args) throws Exception {
    Configuration conf = this.getConf();
    FileSystem fs = FileSystem.get(conf);
    Path tmpPath = new Path("/mp2/tmp");
    fs.delete(tmpPath, true);

    Job jobA = Job.getInstance(conf, "Link Count");
    jobA.setOutputKeyClass(IntWritable.class);
    jobA.setOutputValueClass(IntWritable.class);

    jobA.setMapperClass(LinkCountMap.class);
    jobA.setReducerClass(LinkCountReduce.class);

    FileInputFormat.setInputPaths(jobA, new Path(args[0]));
    FileOutputFormat.setOutputPath(jobA, tmpPath);

    jobA.setJarByClass(TopPopularLinks.class);
    jobA.waitForCompletion(true);

    Job jobB = Job.getInstance(conf, "Top Links");
    jobB.setOutputKeyClass(IntWritable.class);
    jobB.setOutputValueClass(IntWritable.class);

    jobB.setMapOutputKeyClass(NullWritable.class);
    jobB.setMapOutputValueClass(IntArrayWritable.class);

    jobB.setMapperClass(TopLinksMap.class);
    jobB.setReducerClass(TopLinksReduce.class);
    jobB.setNumReduceTasks(1);

    FileInputFormat.setInputPaths(jobB, tmpPath);
    FileOutputFormat.setOutputPath(jobB, new Path(args[1]));

    jobB.setInputFormatClass(KeyValueTextInputFormat.class);
    jobB.setOutputFormatClass(TextOutputFormat.class);

    jobB.setJarByClass(TopPopularLinks.class);
    return jobB.waitForCompletion(true) ? 0 : 1;
  }

  public static class IntArrayWritable extends ArrayWritable {
    public IntArrayWritable() {
      super(IntWritable.class);
    }

    public IntArrayWritable(Integer[] numbers) {
      super(IntWritable.class);
      IntWritable[] ints = new IntWritable[numbers.length];
      for (int i = 0; i < numbers.length; i++) {
        ints[i] = new IntWritable(numbers[i]);
      }
      set(ints);
    }
  }

  public static class LinkCountMap extends Mapper<Object, Text, IntWritable, IntWritable> {

    private static final IntWritable ZERO = new IntWritable(0);
    private static final IntWritable ONE = new IntWritable(1);

    @Override
    public void map(Object key, Text value, Context context)
        throws IOException, InterruptedException {
      String line = value.toString();
      StringTokenizer tokenizer = new StringTokenizer(line, ": ");

      if (tokenizer.hasMoreTokens()) {
        int fromPage = Integer.parseInt(tokenizer.nextToken());
        context.write(new IntWritable(fromPage), ZERO);
        while (tokenizer.hasMoreTokens()) {
          int toPage = Integer.parseInt(tokenizer.nextToken());
          context.write(new IntWritable(toPage), ONE);
        }
      }
    }
  }

  public static class LinkCountReduce
      extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
    @Override
    public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }

      context.write(key, new IntWritable(sum));
    }
  }

  public static class TopLinksMap extends Mapper<Text, Text, NullWritable, IntArrayWritable> {

    private final TreeSet<Pair<Integer, Integer>> linkCountToPageMap = new TreeSet<>();
    private int n;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      this.n = conf.getInt("N", 10);
    }

    @Override
    public void map(Text key, Text value, Context context)
        throws IOException, InterruptedException {
      int pageId = Integer.parseInt(key.toString());
      int linkCount = Integer.parseInt(value.toString());

      linkCountToPageMap.add(new Pair<>(linkCount, pageId));
      if (linkCountToPageMap.size() > n) {
        linkCountToPageMap.remove(linkCountToPageMap.first());
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      for (Pair<Integer, Integer> item : linkCountToPageMap) {
        context.write(
            NullWritable.get(), new IntArrayWritable(new Integer[] {item.second, item.first}));
      }
    }
  }

  public static class TopLinksReduce
      extends Reducer<NullWritable, IntArrayWritable, IntWritable, IntWritable> {

    private final TreeSet<Pair<Integer, Integer>> linkCountToPageMap = new TreeSet<>();
    private int n;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      this.n = conf.getInt("N", 10);
    }

    @Override
    public void reduce(NullWritable key, Iterable<IntArrayWritable> values, Context context)
        throws IOException, InterruptedException {
      for (IntArrayWritable val : values) {
        IntWritable[] pair = (IntWritable[]) val.toArray();
        Integer pageId = Integer.parseInt(pair[0].toString());
        Integer linkCount = Integer.parseInt(pair[1].toString());
        linkCountToPageMap.add(new Pair<>(linkCount, pageId));
        if (linkCountToPageMap.size() > n) {
          linkCountToPageMap.remove(linkCountToPageMap.first());
        }
      }

      for (Pair<Integer, Integer> item : linkCountToPageMap) {
        context.write(new IntWritable(item.second), new IntWritable(item.first));
      }
    }
  }
}

class Pair<A extends Comparable<? super A>, B extends Comparable<? super B>>
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

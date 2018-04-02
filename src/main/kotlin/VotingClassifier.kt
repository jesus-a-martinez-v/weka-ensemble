import weka.classifiers.AbstractClassifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.Logistic
import weka.classifiers.lazy.IBk
import weka.classifiers.meta.AdaBoostM1
import weka.classifiers.meta.Vote
import weka.classifiers.trees.J48
import weka.core.Instances
import weka.core.SelectedTag
import weka.core.converters.CSVLoader
import java.io.File
import java.util.*
import kotlin.math.round

object VotingClassifier {
    fun loadData(full: Boolean = false): Instances {
        fun getDataSetName(): String = if (full) "/bank-full.csv" else "/bank.csv"

        val filePath = this.javaClass.getResource(getDataSetName()).toURI().path
        val file = File(filePath)

        val csvLoader = CSVLoader()
        csvLoader.noHeaderRowPresent = false
        csvLoader.fieldSeparator = ";"
        csvLoader.setSource(file)

        val data = csvLoader.dataSet

        val unknownClassIndex = data.classIndex() == -1

        if (unknownClassIndex) {
            println("Setting class index.")
            data.setClassIndex(data.numAttributes() - 1)
        }

        return data
    }

    fun trainTestSplit(data: Instances, trainProportion: Double = 0.8, seed: Long = 42): Pair<Instances, Instances> {
        require(0 < trainProportion && trainProportion < 1, { "Train proportion must be between 0 and 1." })
        data.randomize(Random(seed))

        val trainSize = round(data.numInstances() * trainProportion).toInt()
        val testSize = data.numInstances() - trainSize

        val train = Instances(data, 0, trainSize)
        val test = Instances(data, trainSize, testSize)

        return Pair(train, test)
    }

    fun trainModel(model: AbstractClassifier, instances: Instances): AbstractClassifier {
        model.buildClassifier(instances)

        return model
    }

    fun evaluateModel(model: AbstractClassifier, train: Instances, test: Instances) {
        val evaluator = Evaluation(train)

        evaluator.evaluateModel(model, test)
        val summary = evaluator.toSummaryString("Results", false)

        println(summary)
    }

    fun createVotingClassifier(): Vote {
        val votingClassifier = Vote()
        val tag = SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES)

        votingClassifier.combinationRule = tag
        votingClassifier.classifiers = getClassifiers()

        return votingClassifier
    }

    fun getClassifiers(): Array<AbstractClassifier> {
        val naiveBayesClassifier = getNaiveBayesClassifier()
        val adaBoostClassifier = getAdaBoostClassifier()
        val logisticRegressionClassifier = getLogisticRegressionClassifier()
        val kNearestNeighborsClassifier = getKNearestNeighborsClassifier()
        val decisionTreeClassifier = getDecisionTreeClassifier()

        return arrayOf(
                naiveBayesClassifier,
                adaBoostClassifier,
                logisticRegressionClassifier,
                kNearestNeighborsClassifier,
                decisionTreeClassifier)
    }

    private fun getNaiveBayesClassifier(): NaiveBayes = NaiveBayes()

    private fun getAdaBoostClassifier(): AdaBoostM1 = AdaBoostM1()

    private fun getLogisticRegressionClassifier(ridge: Double = 1e-8): Logistic {
        val l = Logistic()

        l.ridge = ridge

        return l
    }

    private fun getKNearestNeighborsClassifier(numberOfNeighbors: Int = 3): IBk {
        val knn = IBk()

        knn.knn = numberOfNeighbors

        return knn
    }

    private fun getDecisionTreeClassifier(unpruned: Boolean = true, minObjectsInNode: Int = 2): J48 {
        val decisionTreeClassifier = J48()

        decisionTreeClassifier.unpruned = unpruned
        decisionTreeClassifier.minNumObj = minObjectsInNode

        return decisionTreeClassifier
    }
}

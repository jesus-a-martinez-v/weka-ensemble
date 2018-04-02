import VotingClassifier.createVotingClassifier
import VotingClassifier.evaluateModel
import VotingClassifier.getClassifiers
import VotingClassifier.loadData
import VotingClassifier.trainModel
import VotingClassifier.trainTestSplit
import weka.core.Instances

fun main(args: Array<String>) {
    println("Loading data.")
    val data = loadData(true)

    println("Splitting data.")
    val (train, test) = trainTestSplit(data)

    trainAndEvaluateClassifiersIndividually(train, test)
    trainAndEvaluateVotingClassifier(train, test)
}

fun trainAndEvaluateClassifiersIndividually(train: Instances, test: Instances) {
    getClassifiers().forEach {
        val classifierName = it.javaClass.simpleName
        println("Training: $classifierName")
        trainModel(it, train)
        println("Evaluating: $classifierName")
        evaluateModel(it, train, test)
        println("-------------------------------------")
    }
}

fun trainAndEvaluateVotingClassifier(train: Instances, test: Instances) {
    println("Training voting classifier.")
    val votingClassifier = createVotingClassifier()
    val model = trainModel(votingClassifier, train)

    println("Evaluating model.")
    evaluateModel(model, train, test)
}